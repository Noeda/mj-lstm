// "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN"
// https://doi.org/10.48550/arXiv.1803.04831
//
// A type of recurrent neural network, where the layers don't have self-connections to the same
// layer.
//
// It is also supposed to have a regularizing mechanism that prevents exploding or vanishing
// gradients. but as of 2023-08-29 when I wrote first version and tested it, I did not get
// exploding or vanishing gradients. I may add it later. There's also a bunch of optimization
// things I should do; unnecessary allocations and that sort of thing.
//

use crate::adamw::AdamWConfiguration;
use crate::rnn::{RNNState, RNN};
use crate::simd_common::{
    fast_relu, fast_relu_derivative, fast_sigmoid, fast_sigmoid_derivative, fast_silu,
    fast_silu_derivative, fast_tanh, fast_tanh_derivative,
};
use rand::{thread_rng, Rng};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct AdamWState {
    config: AdamWConfiguration,
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
    iteration: i64,
}

impl AdamWState {
    // Maybe at some point AdamW will not be tied to LSTMv2
    fn new(config: AdamWConfiguration, nn: &IndRNN) -> Self {
        let (vec, _ctx) = nn.to_vec();
        let nparameters = vec.len();
        AdamWState {
            config,
            first_moment: vec![0.0; nparameters],
            second_moment: vec![0.0; nparameters],
            iteration: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct IndRNN {
    weights: Vec<Vec<f64>>,
    u_layers: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    out_weights: Vec<f64>,
    out_biases: Vec<f64>,
    layer_sizes: Vec<usize>,

    activation: IndRNNActivation,
    output_activation: IndRNNActivation,
    freeze_u_layers: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd, Copy)]
pub enum IndRNNActivation {
    LogisticSigmoid,
    Tanh,
    ReLU,
    SiLU,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct IndRNNState {
    activations: Vec<Vec<f64>>,
    timestep: usize,

    state1: Vec<f64>,
    state2: Vec<f64>,

    backprop_steps: Vec<RefCell<BackpropStep>>,
}

unsafe impl Send for IndRNNState {}
unsafe impl Sync for IndRNNState {}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
struct BackpropStep {
    inputs: Vec<f64>,
    last_activations: Vec<Vec<f64>>,
    last_activations_pre_activation: Vec<Vec<f64>>,
    desired_output_derivs: Vec<f64>,
    output_prev_activations: Vec<f64>,
    last_outputs: Vec<f64>,
    last_outputs_pre_activation: Vec<f64>,
    activation_derivs: Vec<Vec<f64>>,
}

impl BackpropStep {
    fn new(nn: &IndRNN, _state: &IndRNNState) -> Self {
        let mut activation_derivs = Vec::new();
        for layer_idx in 1..nn.layer_sizes.len() - 1 {
            activation_derivs.push(vec![0.0; nn.layer_sizes[layer_idx]]);
        }

        Self {
            inputs: vec![],
            last_activations: vec![],
            last_activations_pre_activation: vec![],
            desired_output_derivs: vec![],
            last_outputs: vec![],
            last_outputs_pre_activation: vec![],
            output_prev_activations: vec![],
            activation_derivs,
        }
    }
}

impl Vectorizable for IndRNN {
    type Context = (Vec<usize>, IndRNNActivation, IndRNNActivation, bool);

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut result: Vec<f64> = Vec::new();
        for w in self.weights.iter() {
            result.extend_from_slice(w);
        }
        for u in self.u_layers.iter() {
            result.extend_from_slice(u);
        }
        for b in self.biases.iter() {
            result.extend_from_slice(b);
        }
        for o in self.out_weights.iter() {
            result.push(*o);
        }
        for ob in self.out_biases.iter() {
            result.push(*ob);
        }
        (
            result,
            (
                self.layer_sizes.clone(),
                self.activation,
                self.output_activation,
                self.freeze_u_layers,
            ),
        )
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut rnn = IndRNN::new(&ctx.0);
        let mut vec = vec;
        for w in rnn.weights.iter_mut() {
            let (wvec, rest) = vec.split_at(w.len());
            w.copy_from_slice(wvec);
            vec = rest;
        }
        for u in rnn.u_layers.iter_mut() {
            let (uvec, rest) = vec.split_at(u.len());
            u.copy_from_slice(uvec);
            vec = rest;
        }
        for b in rnn.biases.iter_mut() {
            let (bvec, rest) = vec.split_at(b.len());
            b.copy_from_slice(bvec);
            vec = rest;
        }
        let (out_weights, rest) = vec.split_at(rnn.out_weights.len());
        rnn.out_weights.copy_from_slice(out_weights);
        let (out_biases, _) = rest.split_at(rnn.out_biases.len());
        rnn.out_biases.copy_from_slice(out_biases);
        rnn.activation = ctx.1;
        rnn.output_activation = ctx.2;
        rnn.freeze_u_layers = ctx.3;
        rnn
    }
}

impl IndRNN {
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2);

        let mut weights = Vec::new();
        let mut u_layers = Vec::new();
        let mut biases = Vec::new();

        for idx in 0..layer_sizes.len() - 2 {
            let src_layer_sz = layer_sizes[idx];
            let tgt_layer_sz = layer_sizes[idx + 1];
            let wvec = vec![0.0; src_layer_sz * tgt_layer_sz];
            weights.push(wvec);

            let uvec = vec![0.0; tgt_layer_sz];
            u_layers.push(uvec);

            let bvec = vec![0.0; tgt_layer_sz];
            biases.push(bvec);
        }

        let out_weights =
            vec![0.0; layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2]];

        let out_biases = vec![0.0; layer_sizes[layer_sizes.len() - 1]];

        Self {
            weights,
            u_layers,
            biases,
            out_weights,
            out_biases,
            layer_sizes: layer_sizes.to_vec(),
            activation: IndRNNActivation::LogisticSigmoid,
            output_activation: IndRNNActivation::LogisticSigmoid,
            freeze_u_layers: false,
        }
    }

    /// Freezes u-layer weights.
    ///
    /// Set this on the model that is being optimized (and not on the gradient model).
    ///
    /// The freezing works by not updating parameters for the u-layers when calling backpropagate().
    pub fn set_freeze_u_layers(&mut self, freeze: bool) {
        self.freeze_u_layers = freeze;
    }

    pub fn freeze_u_layers(&self) -> bool {
        self.freeze_u_layers
    }

    /// Sets activation function for hidden layers.
    pub fn set_activation_function(&mut self, activation: IndRNNActivation) {
        self.activation = activation;
    }

    pub fn activation_function(&self) -> IndRNNActivation {
        self.activation
    }

    /// Sets activation function for the output layer. This influences the values you get out in
    /// propagate() (e.g. logistic sigmoid giving values between 0-1, relu giving values between
    /// 0 and infinity, etc.).
    pub fn set_output_activation_function(&mut self, activation: IndRNNActivation) {
        self.output_activation = activation;
    }

    pub fn output_activation_function(&self) -> IndRNNActivation {
        self.output_activation
    }

    pub fn adamw(&self, config: AdamWConfiguration) -> AdamWState {
        AdamWState::new(config, self)
    }

    pub fn clip_weights(&mut self) {
        for u_layer in self.u_layers.iter_mut() {
            for u in u_layer.iter_mut() {
                *u = u.max(-1.0).min(1.0);
            }
        }
    }

    /// Updates gradient based on AdamW optimizer.
    pub fn update_parameters_from_adamw_and_gradient(
        &mut self,
        grad: &Self,
        adamw: &mut AdamWState,
    ) {
        let (mut vec, ctx) = self.to_vec();
        let (gvec, _gctx) = grad.to_vec();
        let nparameters = vec.len();
        assert_eq!(nparameters, gvec.len());
        assert_eq!(nparameters, adamw.first_moment.len());

        let beta1 = adamw.config.beta1;
        let beta2 = adamw.config.beta2;
        let learning_rate = adamw.config.learning_rate;

        for p in 0..adamw.first_moment.len() {
            // first_moment = beta1 * first_moment + (1-beta1) * grad
            let mut beta1_first_moment: f64 = 0.0;
            beta1_first_moment += beta1 * adamw.first_moment[p];
            let mut one_minus_beta1_grad: f64 = 0.0;
            one_minus_beta1_grad += (1.0 - beta1) * gvec[p];
            adamw.first_moment[p] = beta1_first_moment;
            adamw.first_moment[p] += one_minus_beta1_grad;

            // second_moment = beta2 * second_moment + (1-beta2) * grad^2
            let mut beta2_second_moment: f64 = 0.0;
            beta2_second_moment += beta2 * adamw.second_moment[p];
            let mut one_minus_beta2_grad: f64 = 0.0;
            let mut grad2 = gvec[p].clone();
            grad2 *= grad2;
            one_minus_beta2_grad += (1.0 - beta2) * grad2;
            adamw.second_moment[p] = beta2_second_moment;
            adamw.second_moment[p] += one_minus_beta2_grad;

            let iteration: i32 = if adamw.iteration > 1000000 {
                1000000
            } else {
                adamw.iteration as i32
            };

            // bias_correction1 = first_moment / (1 - beta1^t)
            let beta1_t = 1.0 - beta1.powi(iteration);
            let mut bias_correction1 = adamw.first_moment[p];
            bias_correction1 /= beta1_t;

            // bias_correction2 = second_moment / (1 - beta2^t)
            let beta2_t = 1.0 - beta2.powi(iteration);
            let mut bias_correction2 = adamw.second_moment[p];
            bias_correction2 /= beta2_t;

            // adjustment = learning_rate * bias_correction1 / (sqrt(bias_correction2) + epsilon)
            let mut adjustment = learning_rate;
            adjustment *= bias_correction1;
            bias_correction2 = bias_correction2.sqrt();
            bias_correction2 += adamw.config.epsilon;
            adjustment /= bias_correction2;

            let mut decay = adamw.config.weight_decay;
            decay *= vec[p];

            // gradient clip
            if adjustment < -adamw.config.gradient_clip {
                adjustment = -adamw.config.gradient_clip;
            }
            if adjustment > adamw.config.gradient_clip {
                adjustment = adamw.config.gradient_clip;
            }

            // parameter = parameter - adjustment - weight_decay * parameter
            vec[p] -= adjustment;
            vec[p] -= decay;
        }

        *self = Self::from_vec(&vec, &ctx);
        adamw.iteration += 1;
        self.clip_weights();
    }

    pub fn num_inputs(&self) -> usize {
        self.layer_sizes[0]
    }

    pub fn zero_like(&self) -> Self {
        Self::new(&self.layer_sizes)
    }

    pub fn set_zero(&mut self) {
        *self = Self::new(&self.layer_sizes);
    }

    pub fn randomize(&mut self) {
        for w in self.weights.iter_mut() {
            for wval in w.iter_mut() {
                *wval = thread_rng().gen_range(-0.1, 0.1);
            }
        }
        let num_ulayers = self.u_layers.len();
        for (idx, u) in self.u_layers.iter_mut().enumerate() {
            for uval in u.iter_mut() {
                if idx < num_ulayers - 1 {
                    *uval = thread_rng().gen_range(0.0, 1.0);
                }
            }
        }
        for b in self.biases.iter_mut() {
            for bval in b.iter_mut() {
                *bval = thread_rng().gen_range(-0.1, 0.1);
            }
        }
        for o in self.out_weights.iter_mut() {
            *o = thread_rng().gen_range(-0.1, 0.1);
        }
        for ob in self.out_biases.iter_mut() {
            *ob = thread_rng().gen_range(-0.1, 0.1);
        }
    }

    pub fn start_v2(&self) -> IndRNNState {
        let mut activations = Vec::with_capacity(self.layer_sizes.len() - 2);
        let mut widest_layer: usize = self.layer_sizes[0];
        for idx in 0..self.layer_sizes.len() - 1 {
            let u = vec![0.0; self.layer_sizes[idx + 1]];
            activations.push(u);
            if self.layer_sizes[idx + 1] > widest_layer {
                widest_layer = self.layer_sizes[idx + 1];
            }
        }
        IndRNNState {
            activations,
            state1: Vec::with_capacity(widest_layer),
            state2: Vec::with_capacity(widest_layer),
            timestep: 0,
            backprop_steps: vec![],
        }
    }
}

impl RNN for IndRNN {
    type RNNState = (IndRNNState, IndRNN, Vec<f64>);

    fn start(&self) -> Self::RNNState {
        (IndRNN::start_v2(self), self.clone(), vec![])
    }
}

impl RNNState for (IndRNNState, IndRNN, Vec<f64>) {
    type InputType = f64;
    type OutputType = f64;

    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [Self::InputType]) -> &'a [Self::OutputType] {
        self.2
            .resize(self.1.layer_sizes[self.1.layer_sizes.len() - 1], 0.0);
        self.0.propagate_v2(&self.1, inputs, &mut self.2);
        &self.2
    }

    fn propagate32<'a, 'b>(&'a mut self, _inputs: &'b [f32]) -> &'a [f32] {
        unimplemented!();
    }

    fn reset(&mut self) {
        self.0 = IndRNN::start_v2(&self.1);
    }
}

impl IndRNNState {
    pub fn set_output_derivs(&mut self, nn: &IndRNN, output_gradients: &[f64]) {
        assert!(self.backprop_steps.len() > 0);
        assert_eq!(
            output_gradients.len(),
            nn.layer_sizes[nn.layer_sizes.len() - 1]
        );
        let dod = &mut self.backprop_steps[self.backprop_steps.len() - 1]
            .borrow_mut()
            .desired_output_derivs;
        *dod = output_gradients.to_vec();
    }

    pub fn set_output_derivs_for_step(
        &mut self,
        nn: &IndRNN,
        output_gradients: &[f64],
        step: usize,
    ) {
        assert!(self.backprop_steps.len() > 0);
        assert_eq!(
            output_gradients.len(),
            nn.layer_sizes[nn.layer_sizes.len() - 1]
        );
        assert!(step < self.backprop_steps.len());
        let dod = &mut self.backprop_steps[step].borrow_mut().desired_output_derivs;
        *dod = output_gradients.to_vec();
    }

    pub fn get_time_step(&self) -> usize {
        assert!(self.backprop_steps.len() > 0);
        self.backprop_steps.len() - 1
    }

    pub fn propagate_v2(&mut self, rnn: &IndRNN, inputs: &[f64], outputs: &mut Vec<f64>) {
        self.propagate_v2_shadow(rnn, inputs, outputs, false)
    }

    pub fn propagate_collect_gradients_v2(
        &mut self,
        rnn: &IndRNN,
        inputs: &[f64],
        outputs: &mut Vec<f64>,
    ) {
        self.propagate_v2_shadow(rnn, inputs, outputs, true)
    }

    pub fn reset_backpropagation(&mut self) {
        self.backprop_steps.clear();
    }

    pub fn backpropagate(&mut self, nn: &IndRNN, grad: &mut IndRNN) {
        let output_len = nn.layer_sizes[nn.layer_sizes.len() - 1];
        let zeros: Vec<f64> = vec![0.0; output_len];

        let num_backprop_steps = self.backprop_steps.len();

        for step_idx in (0..num_backprop_steps).rev() {
            self.state1.truncate(0);
            self.state2.truncate(0);

            let mut step = self.backprop_steps[step_idx].borrow_mut();
            let mut prev_step = if step_idx > 0 {
                Some(self.backprop_steps[step_idx - 1].borrow_mut())
            } else {
                None
            };
            let ograds: &[f64] = if !step.desired_output_derivs.is_empty() {
                &step.desired_output_derivs
            } else {
                &zeros
            };

            let prev_layer_len = nn.layer_sizes[nn.layer_sizes.len() - 2];
            for target_idx in 0..output_len {
                let output_deriv = match nn.output_activation {
                    IndRNNActivation::LogisticSigmoid => {
                        fast_sigmoid_derivative(step.last_outputs_pre_activation[target_idx])
                    }
                    IndRNNActivation::Tanh => {
                        fast_tanh_derivative(step.last_outputs_pre_activation[target_idx])
                    }
                    IndRNNActivation::ReLU => {
                        fast_relu_derivative(step.last_outputs_pre_activation[target_idx])
                    }
                    IndRNNActivation::SiLU => {
                        fast_silu_derivative(step.last_outputs_pre_activation[target_idx])
                    }
                };
                let output_deriv2 = output_deriv * ograds[target_idx];
                grad.out_biases[target_idx] += output_deriv2;
                self.state2.push(output_deriv2);

                for source_idx in 0..prev_layer_len {
                    grad.out_weights[source_idx + target_idx * prev_layer_len] +=
                        output_deriv2 * step.output_prev_activations[source_idx];
                }
            }
            std::mem::swap(&mut self.state1, &mut self.state2);

            let layer_idx = nn.layer_sizes.len() - 2;

            let num_targets = nn.layer_sizes[layer_idx];
            let num_sources = nn.layer_sizes[layer_idx + 1];

            for target_idx in 0..num_targets {
                let mut act_deriv: f64 = 0.0;
                for source_idx in 0..num_sources {
                    act_deriv += self.state1[source_idx]
                        * nn.out_weights[target_idx + source_idx * num_targets];
                }
                step.activation_derivs[layer_idx - 1][target_idx] += act_deriv;
            }
            std::mem::swap(&mut self.state1, &mut self.state2);

            for layer_idx in (1..nn.layer_sizes.len() - 1).rev() {
                let prev_activations: &[f64] = if layer_idx > 1 {
                    &step.last_activations[layer_idx - 2]
                } else {
                    &step.inputs
                };

                let this_layer_size = nn.layer_sizes[layer_idx];
                let prev_layer_size = nn.layer_sizes[layer_idx - 1];

                let mut activation_derivs_add: Vec<f64> = vec![0.0; prev_layer_size];

                self.state2 = vec![0.0; prev_layer_size];
                for target_idx in 0..this_layer_size {
                    let tld = step.activation_derivs[layer_idx - 1][target_idx];

                    let d_act = match nn.activation {
                        IndRNNActivation::LogisticSigmoid => fast_sigmoid_derivative(
                            step.last_activations_pre_activation[layer_idx - 1][target_idx],
                        ),
                        IndRNNActivation::Tanh => fast_tanh_derivative(
                            step.last_activations_pre_activation[layer_idx - 1][target_idx],
                        ),
                        IndRNNActivation::ReLU => fast_relu_derivative(
                            step.last_activations_pre_activation[layer_idx - 1][target_idx],
                        ),
                        IndRNNActivation::SiLU => fast_silu_derivative(
                            step.last_activations_pre_activation[layer_idx - 1][target_idx],
                        ),
                    };

                    let prev_act: f64 = if let Some(ref mut prev_step) = prev_step {
                        prev_step.last_activations[layer_idx - 1][target_idx]
                    } else {
                        0.0
                    };

                    grad.biases[layer_idx - 1][target_idx] += tld * d_act;
                    if !nn.freeze_u_layers {
                        grad.u_layers[layer_idx - 1][target_idx] += tld * d_act * prev_act;
                    }

                    if let Some(ref mut prev_step) = prev_step {
                        prev_step.activation_derivs[layer_idx - 1][target_idx] +=
                            nn.u_layers[layer_idx - 1][target_idx] * tld * d_act;
                    }

                    for source_idx in 0..prev_layer_size {
                        grad.weights[layer_idx - 1][source_idx + target_idx * prev_layer_size] +=
                            tld * d_act * prev_activations[source_idx];
                        if layer_idx > 1 {
                            activation_derivs_add[source_idx] += nn.weights[layer_idx - 1]
                                [source_idx + target_idx * prev_layer_size]
                                * tld
                                * d_act;
                        }
                    }
                }
                if layer_idx > 1 {
                    for idx in 0..prev_layer_size {
                        step.activation_derivs[layer_idx - 2][idx] += activation_derivs_add[idx];
                    }
                }
                std::mem::swap(&mut self.state1, &mut self.state2);
            }
        }
    }

    fn propagate_v2_shadow(
        &mut self,
        rnn: &IndRNN,
        inputs: &[f64],
        outputs: &mut Vec<f64>,
        collect_gradients: bool,
    ) {
        assert_eq!(inputs.len(), rnn.layer_sizes[0]);
        assert_eq!(outputs.len(), rnn.layer_sizes[rnn.layer_sizes.len() - 1]);

        self.state1.truncate(0);
        self.state1.extend_from_slice(inputs);

        let mut backprop_step: Option<BackpropStep> = None;
        if collect_gradients {
            let mut step = BackpropStep::new(rnn, &self);
            step.inputs.extend_from_slice(inputs);
            backprop_step = Some(step);
        }

        for layer_idx in 0..rnn.layer_sizes.len() - 2 {
            let src_layer_sz = rnn.layer_sizes[layer_idx];
            let tgt_layer_sz = rnn.layer_sizes[layer_idx + 1];

            let biases = &rnn.biases[layer_idx];
            let weights = &rnn.weights[layer_idx];
            let u_layer = &rnn.u_layers[layer_idx];
            let last_activations: &[f64] = &self.activations[layer_idx];
            let mut last_activations_pre_activation: Vec<f64> = vec![];
            if let Some(ref mut _step) = backprop_step {
                last_activations_pre_activation.reserve(tgt_layer_sz);
            }
            self.state2.truncate(0);
            for target_idx in 0..tgt_layer_sz {
                let mut activation: f64 = biases[target_idx];

                // source connections from previous layer
                for source_idx in 0..src_layer_sz {
                    activation +=
                        weights[source_idx + target_idx * src_layer_sz] * self.state1[source_idx];
                }
                activation += u_layer[target_idx] * last_activations[target_idx];
                if let Some(ref mut _step) = backprop_step {
                    last_activations_pre_activation.push(activation);
                }
                activation = match rnn.activation {
                    IndRNNActivation::LogisticSigmoid => fast_sigmoid(activation),
                    IndRNNActivation::Tanh => fast_tanh(activation),
                    IndRNNActivation::ReLU => fast_relu(activation),
                    IndRNNActivation::SiLU => fast_silu(activation),
                };
                self.state2.push(activation);
            }
            self.activations[layer_idx].truncate(0);
            self.activations[layer_idx].extend_from_slice(&self.state2);
            if let Some(ref mut step) = backprop_step {
                step.last_activations.push(self.state2.clone());
                step.last_activations_pre_activation
                    .push(last_activations_pre_activation);
            }
            std::mem::swap(&mut self.state1, &mut self.state2);
        }

        // output linear layer
        if let Some(ref mut step) = backprop_step {
            step.last_outputs_pre_activation.truncate(0);
        }
        let outputs_len = outputs.len();
        for target_idx in 0..outputs_len {
            let mut activation: f64 = rnn.out_biases[target_idx];
            let ninputs = self.state1.len();
            for source_idx in 0..ninputs {
                activation +=
                    rnn.out_weights[source_idx + target_idx * ninputs] * self.state1[source_idx];
            }
            if let Some(ref mut step) = backprop_step {
                step.last_outputs_pre_activation.push(activation);
            }
            outputs[target_idx] = match rnn.output_activation {
                IndRNNActivation::LogisticSigmoid => fast_sigmoid(activation),
                IndRNNActivation::Tanh => fast_tanh(activation),
                IndRNNActivation::ReLU => fast_relu(activation),
                IndRNNActivation::SiLU => fast_silu(activation),
            };
        }
        if let Some(ref mut step) = backprop_step {
            step.output_prev_activations.extend_from_slice(&self.state1);
            step.last_outputs.extend_from_slice(outputs);
            self.backprop_steps
                .push(RefCell::new(backprop_step.unwrap()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    pub fn to_vec_and_from_vec_is_id() {
        let mut rnn = IndRNN::new(&[10, 20, 30, 20, 7]);
        rnn.randomize();

        let (vec, ctx) = rnn.to_vec();
        let rnn2 = IndRNN::from_vec(&vec, &ctx);
        let (vec2, _ctx2) = rnn2.to_vec();

        assert_eq!(vec.len(), vec2.len());
        for v in 0..vec.len() {
            assert!((vec[v] - vec2[v]).abs() < 0.0001);
        }
    }

    #[test]
    pub fn haskell_comparison() {
        let mut rnn = IndRNN::new(&[2, 4, 3, 2]);
        rnn.randomize();

        // read from hs2/model.json and replace the weights in rnn with them
        let mut rnn_json = String::new();
        let mut f = std::fs::File::open("hs2/model.json").unwrap();
        f.read_to_string(&mut rnn_json).unwrap();
        let rnn: IndRNN = serde_json::from_str(&rnn_json).unwrap();

        let mut inp: Vec<f64> = vec![1.2, 2.4];
        let mut st = rnn.start_v2();
        let mut out: Vec<f64> = vec![0.0; 2];
        st.propagate_collect_gradients_v2(&rnn, &inp, &mut out);
        st.set_output_derivs(&rnn, &[1.0, 1.0]);
        inp[0] = 1.3;
        inp[1] = 2.5;
        st.propagate_collect_gradients_v2(&rnn, &inp, &mut out);
        st.set_output_derivs(&rnn, &[1.0, 1.0]);
        inp[0] = 1.7;
        inp[1] = -2.0;
        st.propagate_collect_gradients_v2(&rnn, &inp, &mut out);
        st.set_output_derivs(&rnn, &[1.0, 1.0]);
        let mut grad = rnn.zero_like();
        st.backpropagate(&rnn, &mut grad);
        println!("out={:?}", out);

        let rnn_json = serde_json::to_string(&rnn).unwrap();
        println!("{:?}", grad);
        //println!("{}\n", rnn_json);
    }
}
