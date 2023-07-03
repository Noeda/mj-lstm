/*
 * Re-implementation of lstm.rs to be a bit more clear and fix some issues.
 * It is not compatible with the lstm.rs implementation because the original lacks
 * LSTM self-connections due to oversight.
 *
 * It implements an architecture with N inputs, a linear connection to the first LSTM layer,
 * then M LSTM layers, and then linear connections to an output layer.
 *
 * The parameters include biases for output layer. This version applies sigmoid function at the last layer
 * to be consistent with lstm.rs
 */

// Notes about representation
//
// All parameters are in a single Vec<F64x4>.
//
// For LSTM this works nicely, because input, input gate, forget gate and output gate require exactly 4 parameters.
//
// The parameters consist of:
//   * The weight parameters between all layers
//   * The self-connecting weights on each LSTM layer
//   * Biases on each LSTM layer and output layer
//
// 2023-07-02: First version where gradient calculation does not seem obviously off, verified with
// Haskell. Not sure everything continues to be good if we try more complicated scenarios.

use crate::rnn::{RNNState, RNN};
use crate::simd_common::{
    fast_sigmoid, fast_sigmoid_derivative, fast_tanh, fast_tanh_derivative, inv_fast_sigmoid,
    inv_fast_tanh, F64x4,
};
use rand::{thread_rng, Rng};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct LSTMv2 {
    parameters: Vec<F64x4>,
    layer_sizes: Vec<usize>,
}

impl Vectorizable for LSTMv2 {
    type Context = Vec<usize>;

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut result: Vec<f64> = vec![];
        for p4 in self.parameters.iter() {
            result.push(p4.v1());
            result.push(p4.v2());
            result.push(p4.v3());
            result.push(p4.v4());
        }
        (result, self.layer_sizes.clone())
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut result = LSTMv2::new(ctx);
        let mut cursor: usize = 0;
        for p4 in result.parameters.iter_mut() {
            *p4 = F64x4::new(
                vec[cursor],
                vec[cursor + 1],
                vec[cursor + 2],
                vec[cursor + 3],
            );
            cursor += 4;
        }
        result
    }
}

impl RNN for LSTMv2 {
    type RNNState = (LSTMv2State, LSTMv2, Vec<f64>);

    fn start(&self) -> Self::RNNState {
        (LSTMv2::start_v2(self), self.clone(), vec![])
    }
}

#[derive(Clone, Debug)]
pub struct LSTMv2State {
    memories: Vec<f64>,
    state1: Vec<f64>,
    state2: Vec<f64>,
    state3: Vec<F64x4>,

    last_activations: Vec<f64>,

    // yep it's just a vector of zeros
    zeros: Vec<f64>,

    // for backpropagation.
    backprop_steps: Vec<RefCell<BackpropStep>>,
}

// RefCell is technically not thread-safe, but all methods for LSTMv2State take a &mut so multiple
// thread access is prevented that way.
unsafe impl Send for LSTMv2State {}
unsafe impl Sync for LSTMv2State {}

#[derive(Clone, Debug)]
pub struct AdamWState {
    config: AdamWConfiguration,
    first_moment: Vec<F64x4>,
    second_moment: Vec<F64x4>,
    iteration: i64,
}

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct AdamWConfiguration {
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    weight_decay: f64,
    gradient_clip: f64,
}

impl AdamWConfiguration {
    pub fn new() -> Self {
        AdamWConfiguration {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            learning_rate: 0.001,
            weight_decay: 0.0,
            gradient_clip: std::f64::INFINITY,
        }
    }

    pub fn gradient_clip(self, gradient_clip: f64) -> Self {
        Self {
            gradient_clip,
            ..self
        }
    }
}

impl Default for AdamWConfiguration {
    fn default() -> Self {
        Self::new()
    }
}

impl AdamWState {
    // Maybe at some point AdamW will not be tied to LSTMv2
    fn new(config: AdamWConfiguration, nn: &LSTMv2) -> Self {
        let nparameters = nn.parameters.len();
        AdamWState {
            config,
            first_moment: vec![F64x4::new(0.0, 0.0, 0.0, 0.0); nparameters],
            second_moment: vec![F64x4::new(0.0, 0.0, 0.0, 0.0); nparameters],
            iteration: 1,
        }
    }
}

#[derive(Clone, Debug)]
struct BackpropStep {
    last_memories: Vec<f64>,
    memories: Vec<f64>,
    last_outputs: Vec<f64>,
    last_activations: Vec<f64>,
    gate_activations: Vec<F64x4>,
    inputs: Vec<f64>,
    output_prev_activations: Vec<f64>,
    activation_derivs: Vec<f64>,

    desired_output_derivs: Option<Vec<f64>>,
}

impl BackpropStep {
    fn new(st: &LSTMv2State, layer_sizes: &[usize], ninputs: usize, noutputs: usize) -> Self {
        let gate_activations = vec![F64x4::new(0.0, 0.0, 0.0, 0.0); st.memories.len()];
        let last_memories = vec![0.0; st.memories.len()];
        let memories = vec![0.0; st.memories.len()];
        let last_outputs = vec![0.0; noutputs];
        let inputs = vec![0.0; ninputs];
        let last_activations = vec![0.0; st.memories.len()];
        let output_prev_activations = vec![
            0.0;
            if last_memories.len() == 0 {
                ninputs
            } else {
                layer_sizes[layer_sizes.len() - 2]
            }
        ];
        let activation_derivs = vec![0.0; st.memories.len()];
        BackpropStep {
            last_memories,
            memories,
            last_outputs,
            last_activations,
            gate_activations,
            inputs,
            output_prev_activations,
            activation_derivs,
            desired_output_derivs: None,
        }
    }
}

impl RNNState for (LSTMv2State, LSTMv2, Vec<f64>) {
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
        self.0 = LSTMv2::start_v2(&self.1);
    }
}

fn round_to_4(x: usize) -> usize {
    if x % 4 == 0 {
        x
    } else {
        x + 4 - (x % 4)
    }
}

// These functions compute how many parameters certain weight or biases need.
// If you see a * 4, then that means it's for each of the LSTM connections
// (input, input gate, output gate, forget gate)
fn layer_to_layer_nparameters(num_inputs: usize, num_hiddens: usize) -> usize {
    num_inputs * num_hiddens * 4
}

fn layer_biases_nparameters(num_hidden: usize) -> usize {
    num_hidden * 4
}

fn layer_to_output_nparameters(num_inputs: usize, num_outputs: usize) -> usize {
    round_to_4(num_inputs * num_outputs)
}

fn output_biases_nparameters(num_outputs: usize) -> usize {
    round_to_4(num_outputs)
}

// Counts how many parameters exist between the given layers.
// end_idx is not included in the range, so start_idx == end_idx will cause this function to return 0.
fn count_parameters(layer_sizes: &[usize], start_idx: usize, end_idx: usize) -> usize {
    let mut num_parameters: usize = 0;
    for layer_idx in start_idx..end_idx {
        // Weights from previous layer to this layer
        num_parameters +=
            layer_to_layer_nparameters(layer_sizes[layer_idx - 1], layer_sizes[layer_idx]);
        // LSTM layer to itself
        num_parameters +=
            layer_to_layer_nparameters(layer_sizes[layer_idx], layer_sizes[layer_idx]);
        // Biases of LSTM layers
        num_parameters += layer_biases_nparameters(layer_sizes[layer_idx]);
    }
    num_parameters
}

impl LSTMv2 {
    /// Creates a new LSTMv2 with the given layer sizes.
    ///
    /// All parameters are initialized to 0. Use .randomize() to initialize to random values.
    /// All-zero LSTMv2 won't have any gradients.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "LSTMV2 requires at least 2 layers, for input and output layer"
        );

        let num_outputs: usize = *layer_sizes.last().unwrap();

        let mut num_parameters: usize = 0;
        // Parameters between the hidden layers
        num_parameters += count_parameters(layer_sizes, 1, layer_sizes.len() - 1);

        // Last LSTM layer to output layer
        num_parameters +=
            layer_to_output_nparameters(layer_sizes[layer_sizes.len() - 2], num_outputs);
        // Biases for output layer
        num_parameters += output_biases_nparameters(num_outputs);

        let parameters = vec![F64x4::new(0.0, 0.0, 0.0, 0.0); (num_parameters + 3) / 4];

        Self {
            layer_sizes: layer_sizes.to_vec(),
            parameters,
        }
    }

    /// Create AdamW optimizer for this LSTMv2.
    pub fn adamw(&self, config: AdamWConfiguration) -> AdamWState {
        AdamWState::new(config, self)
    }

    pub fn randomize(&mut self) {
        let mut rng = thread_rng();
        for p in &mut self.parameters {
            *p = F64x4::new(
                rng.gen_range(-0.5, 0.5),
                rng.gen_range(-0.5, 0.5),
                rng.gen_range(-0.5, 0.5),
                rng.gen_range(-0.5, 0.5),
            );
        }
    }

    /// Nudges every parameter in the network in the negative direction of the given gradient,
    /// given a learning rate.
    ///
    /// You can use this to implement basic Stochastic Gradient Descent.
    pub fn update_parameters_from_gradient(&mut self, grad: &Self, learning_rate: f64) {
        assert_eq!(self.parameters.len(), grad.parameters.len());
        assert_eq!(self.layer_sizes, grad.layer_sizes);

        for idx in 0..self.parameters.len() {
            self.parameters[idx].mul_add_scalar(-learning_rate, grad.parameters[idx]);
        }
    }

    /// Updates gradient based on AdamW optimizer.
    pub fn update_parameters_from_adamw_and_gradient(
        &mut self,
        grad: &Self,
        adamw: &mut AdamWState,
    ) {
        assert_eq!(self.parameters.len(), grad.parameters.len());
        assert_eq!(self.parameters.len(), adamw.first_moment.len());

        let beta1 = adamw.config.beta1;
        let beta2 = adamw.config.beta2;
        let learning_rate = adamw.config.learning_rate;

        for p in 0..adamw.first_moment.len() {
            // first_moment = beta1 * first_moment + (1-beta1) * grad
            let mut beta1_first_moment = F64x4::new(0.0, 0.0, 0.0, 0.0);
            beta1_first_moment.mul_add_scalar(adamw.config.beta1, adamw.first_moment[p]);
            let mut one_minus_beta1_grad = F64x4::new(0.0, 0.0, 0.0, 0.0);
            one_minus_beta1_grad.mul_add_scalar(1.0 - adamw.config.beta1, grad.parameters[p]);
            adamw.first_moment[p] = beta1_first_moment;
            adamw.first_moment[p].add(one_minus_beta1_grad);

            // second_moment = beta2 * second_moment + (1-beta2) * grad^2
            let mut beta2_second_moment = F64x4::new(0.0, 0.0, 0.0, 0.0);
            beta2_second_moment.mul_add_scalar(adamw.config.beta2, adamw.second_moment[p]);
            let mut one_minus_beta2_grad = F64x4::new(0.0, 0.0, 0.0, 0.0);
            let mut grad2 = grad.parameters[p].clone();
            grad2.mul(grad2);
            one_minus_beta2_grad.mul_add_scalar(1.0 - adamw.config.beta2, grad2);
            adamw.second_moment[p] = beta2_second_moment;
            adamw.second_moment[p].add(one_minus_beta2_grad);

            let iteration: i32 = if adamw.iteration > 1000000 {
                1000000
            } else {
                adamw.iteration as i32
            };

            // bias_correction1 = first_moment / (1 - beta1^t)
            let beta1_t = 1.0 - beta1.powi(iteration);
            let mut bias_correction1 = adamw.first_moment[p];
            bias_correction1.div_scalar(beta1_t);

            // bias_correction2 = second_moment / (1 - beta2^t)
            let beta2_t = 1.0 - beta2.powi(iteration);
            let mut bias_correction2 = adamw.second_moment[p];
            bias_correction2.div_scalar(beta2_t);

            // adjustment = learning_rate * bias_correction1 / (sqrt(bias_correction2) + epsilon)
            let mut adjustment = F64x4::broadcast(learning_rate);
            adjustment.mul(bias_correction1);
            bias_correction2.sqrt();
            bias_correction2.add_scalar(adamw.config.epsilon);
            adjustment.div(bias_correction2);

            let mut decay = F64x4::broadcast(adamw.config.weight_decay);
            decay.mul(self.parameters[p]);

            // gradient clip
            adjustment.min_scalar(adamw.config.gradient_clip, adjustment);
            adjustment.max_scalar(-adamw.config.gradient_clip, adjustment);

            // parameter = parameter - adjustment - weight_decay * parameter
            self.parameters[p].sub(adjustment);
            self.parameters[p].sub(decay);
        }
    }

    /// Sets every parameter in the LSTM to zero.
    ///
    /// Useful to reset gradients if the LSTMv2 is used as gradient accumulator.
    pub fn set_zero(&mut self) {
        for param in &mut self.parameters {
            *param = F64x4::new(0.0, 0.0, 0.0, 0.0);
        }
    }

    /// Creates a new LSTMv2 with the same shape but all zero parameters.
    pub fn zero_like(&self) -> Self {
        let new = Self::new(&self.layer_sizes);
        new
    }

    pub fn num_inputs(&self) -> usize {
        self.layer_sizes[0]
    }

    pub fn num_outputs(&self) -> usize {
        *self.layer_sizes.last().unwrap()
    }

    pub fn start_v2(&self) -> LSTMv2State {
        let memories: Vec<f64> = vec![0.0; self.count_memory_cells()];
        let memories_len = memories.len();
        LSTMv2State {
            memories,
            last_activations: vec![0.0; memories_len],
            backprop_steps: vec![],
            state1: vec![0.0; self.widest_layer()],
            state2: vec![0.0; self.widest_layer()],
            state3: vec![F64x4::new(0.0, 0.0, 0.0, 0.0); (self.widest_layer() + 3) / 4],
            zeros: vec![0.0; self.widest_layer()],
        }
    }

    /// Returns the number of cells in the widest cell in the network.
    fn widest_layer(&self) -> usize {
        *self.layer_sizes.iter().max().unwrap()
    }

    /// Counts how many memory cells exist for storing LSTMv2 state
    fn count_memory_cells(&self) -> usize {
        let mut result: usize = 0;
        for i in 1..self.layer_sizes.len() - 1 {
            result += self.layer_sizes[i];
        }
        result
    }

    fn layer_to_layer_parameters(&self, layer_idx: usize) -> &[F64x4] {
        assert!(layer_idx > 0);
        assert!(layer_idx < self.layer_sizes.len() - 1);
        assert!(self.layer_sizes.len() > 2);

        let cursor: usize = count_parameters(&self.layer_sizes, 1, layer_idx) / 4;
        &self.parameters[cursor
            ..cursor
                + layer_to_layer_nparameters(
                    self.layer_sizes[layer_idx - 1],
                    self.layer_sizes[layer_idx],
                ) / 4]
    }

    /// Gives a slice parameters from previous layer to given layer.
    /// 0 is the input layer. 1 is considered to be the first LSTM layer.
    fn layer_to_layer_parameters_mut(&mut self, layer_idx: usize) -> &mut [F64x4] {
        assert!(layer_idx > 0);
        assert!(layer_idx < self.layer_sizes.len() - 1);
        assert!(self.layer_sizes.len() > 2);

        let cursor: usize = count_parameters(&self.layer_sizes, 1, layer_idx) / 4;
        &mut self.parameters[cursor
            ..cursor
                + layer_to_layer_nparameters(
                    self.layer_sizes[layer_idx - 1],
                    self.layer_sizes[layer_idx],
                ) / 4]
    }

    /// Gives a slice to LSTM parameters that connect to itself
    fn layer_to_self_parameters(&self, layer_idx: usize) -> &[F64x4] {
        assert!(layer_idx > 0);
        assert!(layer_idx < self.layer_sizes.len() - 1);
        assert!(self.layer_sizes.len() > 2);

        let mut cursor: usize = count_parameters(&self.layer_sizes, 1, layer_idx) / 4;
        // Skip over layer to layer parameters
        cursor += layer_to_layer_nparameters(
            self.layer_sizes[layer_idx - 1],
            self.layer_sizes[layer_idx],
        ) / 4;
        &self.parameters[cursor
            ..cursor
                + layer_to_layer_nparameters(
                    self.layer_sizes[layer_idx],
                    self.layer_sizes[layer_idx],
                ) / 4]
    }

    /// Gives a slice to LSTM parameters that connect to itself
    fn layer_to_self_parameters_mut(&mut self, layer_idx: usize) -> &mut [F64x4] {
        assert!(layer_idx > 0);
        assert!(layer_idx < self.layer_sizes.len() - 1);
        assert!(self.layer_sizes.len() > 2);

        let mut cursor: usize = count_parameters(&self.layer_sizes, 1, layer_idx) / 4;
        // Skip over layer to layer parameters
        cursor += layer_to_layer_nparameters(
            self.layer_sizes[layer_idx - 1],
            self.layer_sizes[layer_idx],
        ) / 4;
        &mut self.parameters[cursor
            ..cursor
                + layer_to_layer_nparameters(
                    self.layer_sizes[layer_idx],
                    self.layer_sizes[layer_idx],
                ) / 4]
    }

    /// Gives a slice to the biases on an LSTM layer
    fn layer_bias_parameters(&self, layer_idx: usize) -> &[F64x4] {
        assert!(layer_idx > 0);
        assert!(layer_idx < self.layer_sizes.len() - 1);
        assert!(self.layer_sizes.len() > 2);

        let mut cursor: usize = count_parameters(&self.layer_sizes, 1, layer_idx) / 4;
        // Skip over layer to layer parameters
        cursor += layer_to_layer_nparameters(
            self.layer_sizes[layer_idx - 1],
            self.layer_sizes[layer_idx],
        ) / 4;
        // Skip over layer to self parameters
        cursor +=
            layer_to_layer_nparameters(self.layer_sizes[layer_idx], self.layer_sizes[layer_idx])
                / 4;
        &self.parameters[cursor..cursor + layer_biases_nparameters(self.layer_sizes[layer_idx]) / 4]
    }

    fn layer_bias_parameters_mut(&mut self, layer_idx: usize) -> &mut [F64x4] {
        assert!(layer_idx > 0);
        assert!(layer_idx < self.layer_sizes.len() - 1);
        assert!(self.layer_sizes.len() > 2);

        let mut cursor: usize = count_parameters(&self.layer_sizes, 1, layer_idx) / 4;
        // Skip over layer to layer parameters
        cursor += layer_to_layer_nparameters(
            self.layer_sizes[layer_idx - 1],
            self.layer_sizes[layer_idx],
        ) / 4;
        // Skip over layer to self parameters
        cursor +=
            layer_to_layer_nparameters(self.layer_sizes[layer_idx], self.layer_sizes[layer_idx])
                / 4;
        &mut self.parameters
            [cursor..cursor + layer_biases_nparameters(self.layer_sizes[layer_idx]) / 4]
    }

    /// Slice to last LSTM layer (or input layer if there are no LSTM layers) to output layer parameters
    fn layer_to_output_parameters(&self) -> &[f64] {
        self.layer_to_output_parameters_mut()
    }

    fn layer_to_output_parameters_mut(&self) -> &mut [f64] {
        let cursor: usize = count_parameters(&self.layer_sizes, 1, self.layer_sizes.len() - 1) / 4;
        let result: &[F64x4] = &self.parameters[cursor
            ..cursor
                + layer_to_output_nparameters(
                    self.layer_sizes[self.layer_sizes.len() - 2],
                    self.layer_sizes[self.layer_sizes.len() - 1],
                ) / 4];
        // Return a slice with the proper size, without padding to a number divisible by 4
        unsafe {
            std::slice::from_raw_parts_mut(
                result.as_ptr() as *mut f64,
                self.layer_sizes[self.layer_sizes.len() - 2]
                    * self.layer_sizes[self.layer_sizes.len() - 1],
            )
        }
    }

    fn output_biases_parameters(&self) -> &[f64] {
        let mut cursor: usize =
            count_parameters(&self.layer_sizes, 1, self.layer_sizes.len() - 1) / 4;
        cursor += layer_to_output_nparameters(
            self.layer_sizes[self.layer_sizes.len() - 2],
            self.layer_sizes[self.layer_sizes.len() - 1],
        ) / 4;
        let result: &[F64x4] = &self.parameters[cursor
            ..cursor + output_biases_nparameters(self.layer_sizes[self.layer_sizes.len() - 1]) / 4];
        // Return a slice with the proper size, without padding to a number divisible by 4
        unsafe {
            std::slice::from_raw_parts(
                result.as_ptr() as *const f64,
                self.layer_sizes[self.layer_sizes.len() - 1],
            )
        }
    }

    fn output_biases_parameters_mut(&self) -> &mut [f64] {
        let mut cursor: usize =
            count_parameters(&self.layer_sizes, 1, self.layer_sizes.len() - 1) / 4;
        cursor += layer_to_output_nparameters(
            self.layer_sizes[self.layer_sizes.len() - 2],
            self.layer_sizes[self.layer_sizes.len() - 1],
        ) / 4;
        let result: &[F64x4] = &self.parameters[cursor
            ..cursor + output_biases_nparameters(self.layer_sizes[self.layer_sizes.len() - 1]) / 4];
        // Return a slice with the proper size, without padding to a number divisible by 4
        unsafe {
            std::slice::from_raw_parts_mut(
                result.as_ptr() as *mut f64,
                self.layer_sizes[self.layer_sizes.len() - 1],
            )
        }
    }
}

impl LSTMv2State {
    pub fn propagate_v2(&mut self, nn: &LSTMv2, inputs: &[f64], outputs: &mut Vec<f64>) {
        self.propagate_v2_shadow(nn, inputs, outputs, false);
    }

    pub fn propagate_collect_gradients_v2(
        &mut self,
        nn: &LSTMv2,
        inputs: &[f64],
        outputs: &mut Vec<f64>,
    ) {
        self.propagate_v2_shadow(nn, inputs, outputs, true);
    }

    /// Sets derivatives for outputs. Call this after you've called propagate_collect_gradients_v2
    /// to tell the network what your "desired" output was.
    ///
    /// You can omit calling this on time steps, in those cases the outputs will not contribute to
    /// gradients. (I.e. it's like the training will not care what the network outputted).
    pub fn set_output_derivs(&mut self, nn: &LSTMv2, output_gradients: &[f64]) {
        assert!(self.backprop_steps.len() > 0);
        assert_eq!(
            output_gradients.len(),
            nn.layer_sizes[nn.layer_sizes.len() - 1]
        );
        let dod = &mut self.backprop_steps[self.backprop_steps.len() - 1]
            .borrow_mut()
            .desired_output_derivs;
        *dod = Some(output_gradients.to_vec());
    }

    /// Resets all backpropagation state. If you are re-using the state then you'll want to call
    /// this after you've adjusted the gradients or it'll re-use your old gradients.
    pub fn reset_backpropagation(&mut self) {
        self.backprop_steps.clear();
    }

    /// Runs backpropagation through time and adds the gradients to given 'grad'.
    ///
    /// 'grad' is not cleared before calling this (to allow accumulation of gradients).
    ///
    /// Also, backpropagation state is not cleared so if you call this again, the same gradients
    /// are accumulated again. Use reset_backpropagation() if you want to clear the state.
    pub fn backpropagate(&mut self, nn: &LSTMv2, grad: &mut LSTMv2) {
        let mut full_state_offset: usize = 0;
        for layer_idx in 1..nn.layer_sizes.len() - 1 {
            full_state_offset += nn.layer_sizes[layer_idx];
        }

        let num_backprop_steps = self.backprop_steps.len();

        for step_idx in (0..num_backprop_steps).rev() {
            let mut step = self.backprop_steps[step_idx].borrow_mut();
            // derivatives are sent to previous step as well
            let mut prev_step = if step_idx > 0 {
                Some(self.backprop_steps[step_idx - 1].borrow_mut())
            } else {
                None
            };

            let ograds: &[f64] = match step.desired_output_derivs {
                Some(ref derivs) => derivs,
                None => &self.zeros,
            };

            let output_len: usize = nn.layer_sizes[nn.layer_sizes.len() - 1];
            let g_output_biases: &mut [f64] = &mut grad.output_biases_parameters_mut();

            let prev_layer_len = nn.layer_sizes[nn.layer_sizes.len() - 2];
            for target_idx in 0..output_len {
                let output_deriv =
                    fast_sigmoid_derivative(inv_fast_sigmoid(step.last_outputs[target_idx]));
                let output_deriv2 = output_deriv * ograds[target_idx];
                g_output_biases[target_idx] += output_deriv2;
                self.state2[target_idx] = output_deriv2;

                for source_idx in 0..prev_layer_len {
                    let g_output_wgt: &mut f64 = &mut grad.layer_to_output_parameters_mut()
                        [source_idx + target_idx * prev_layer_len];
                    *g_output_wgt += ograds[target_idx]
                        * output_deriv
                        * step.output_prev_activations[source_idx];
                }
            }
            std::mem::swap(&mut self.state1, &mut self.state2);

            if nn.layer_sizes.len() > 2 {
                {
                    let layer_idx = nn.layer_sizes.len() - 2;

                    let num_targets = nn.layer_sizes[layer_idx];
                    let num_sources = nn.layer_sizes[layer_idx + 1];

                    let lower_layer_derivs: &[f64] = &self.state1[0..num_sources];
                    let this_layer_derivs: &mut [f64] = &mut step.activation_derivs
                        [full_state_offset - num_targets..full_state_offset];

                    let wgts = nn.layer_to_output_parameters();
                    for target_idx in 0..num_targets {
                        let mut act_deriv: f64 = 0.0;
                        for source_idx in 0..num_sources {
                            act_deriv += lower_layer_derivs[source_idx]
                                * wgts[target_idx + source_idx * num_targets];
                        }
                        this_layer_derivs[target_idx] += act_deriv;
                    }
                }

                let mut state_offset = full_state_offset;
                // derivatives up the hidden layer chain.
                for layer_idx in (1..nn.layer_sizes.len() - 1).rev() {
                    let this_layer_size = nn.layer_sizes[layer_idx];
                    let prev_layer_size = nn.layer_sizes[layer_idx - 1];
                    state_offset -= this_layer_size;
                    let gate_acts: &[F64x4] =
                        &step.gate_activations[state_offset..state_offset + this_layer_size];
                    let memories = &step.memories[state_offset..state_offset + this_layer_size];

                    let last_memories =
                        &step.last_memories[state_offset..state_offset + this_layer_size];
                    let activations =
                        &step.last_activations[state_offset..state_offset + this_layer_size];
                    let prev_activations: &[f64] = if layer_idx > 1 {
                        &step.last_activations[state_offset - prev_layer_size
                            ..state_offset - prev_layer_size + prev_layer_size]
                    } else {
                        &step.inputs[0..prev_layer_size]
                    };

                    // derivatives for the outputs of the current layer
                    let this_layer_derivs: &[f64] =
                        &step.activation_derivs[state_offset..state_offset + this_layer_size];
                    // derivatives for the outputs of the previous layer
                    // unsafe code to take slice borrows that don't overlap
                    // we do this by forcing immutable to mut.
                    //
                    // Rust says undefined behavior but so far it's never done the wrong thing so piss off
                    let mut lower_layer_derivs: Option<&mut [f64]> = unsafe {
                        if layer_idx > 1 {
                            let ptr: *const f64 = step.activation_derivs
                                [state_offset - prev_layer_size..state_offset]
                                .as_ptr();
                            let ptr: *mut f64 = ptr as *mut f64;
                            Some(std::slice::from_raw_parts_mut(ptr, prev_layer_size))
                        } else {
                            None
                        }
                    };

                    for target_idx in 0..this_layer_size {
                        let tld = this_layer_derivs[target_idx];

                        let d_input =
                            fast_tanh_derivative(inv_fast_tanh(gate_acts[target_idx].v1()));
                        let d_input_gate =
                            fast_sigmoid_derivative(inv_fast_sigmoid(gate_acts[target_idx].v2()));
                        let d_output_gate =
                            fast_sigmoid_derivative(inv_fast_sigmoid(gate_acts[target_idx].v3()));
                        let d_forget_gate =
                            fast_sigmoid_derivative(inv_fast_sigmoid(gate_acts[target_idx].v4()));

                        let mut deriv_new_memory = tld
                            * gate_acts[target_idx].v3()
                            * fast_tanh_derivative(memories[target_idx]);
                        if step_idx < num_backprop_steps - 1 {
                            let future_step = self.backprop_steps[step_idx + 1].borrow();
                            let future_gate_acts: &[F64x4] = &future_step.gate_activations
                                [state_offset..state_offset + this_layer_size];
                            let future_this_layer_derivs: &[f64] = &future_step.activation_derivs
                                [state_offset..state_offset + this_layer_size];
                            let future_memories: &[f64] =
                                &future_step.memories[state_offset..state_offset + this_layer_size];
                            let future_deriv_new_memory = future_this_layer_derivs[target_idx]
                                * future_gate_acts[target_idx].v3()
                                * fast_tanh_derivative(future_memories[target_idx]);

                            deriv_new_memory +=
                                future_gate_acts[target_idx].v4() * future_deriv_new_memory;
                        }
                        let deriv_input = deriv_new_memory * gate_acts[target_idx].v2() * d_input;
                        let deriv_input_gate =
                            deriv_new_memory * gate_acts[target_idx].v1() * d_input_gate;
                        let deriv_output_gate =
                            tld * fast_tanh(memories[target_idx]) * d_output_gate;

                        let deriv_forget_gate =
                            deriv_new_memory * last_memories[target_idx] * d_forget_gate;

                        let deriv: F64x4 = F64x4::new(
                            deriv_input,
                            deriv_input_gate,
                            deriv_output_gate,
                            deriv_forget_gate,
                        );

                        let g_wgts: &mut [F64x4] = grad.layer_to_layer_parameters_mut(layer_idx);
                        let wgts: &[F64x4] = &nn.layer_to_layer_parameters(layer_idx);
                        for source_idx in 0..prev_layer_size {
                            g_wgts[source_idx + target_idx * prev_layer_size]
                                .mul_add_scalar(prev_activations[source_idx], deriv);
                            let add_deriv = deriv_input
                                * wgts[source_idx + target_idx * prev_layer_size].v1()
                                + deriv_input_gate
                                    * wgts[source_idx + target_idx * prev_layer_size].v2()
                                + deriv_output_gate
                                    * wgts[source_idx + target_idx * prev_layer_size].v3()
                                + deriv_forget_gate
                                    * wgts[source_idx + target_idx * prev_layer_size].v4();
                            if let Some(ref mut lld) = lower_layer_derivs {
                                lld[source_idx] += add_deriv;
                            }
                        }
                        // self connections
                        let g_self_wgts: &mut [F64x4] =
                            grad.layer_to_self_parameters_mut(layer_idx);
                        let self_wgts: &[F64x4] = &nn.layer_to_self_parameters(layer_idx);
                        for source_idx in 0..this_layer_size {
                            g_self_wgts[source_idx + target_idx * this_layer_size]
                                .mul_add_scalar(activations[source_idx], deriv);
                            if let Some(ref mut lld) = prev_step {
                                let prev_step_this_layer_derivs: &mut [f64] = &mut lld
                                    .activation_derivs
                                    [state_offset..state_offset + this_layer_size];
                                let add_deriv = deriv_input
                                    * self_wgts[source_idx + target_idx * this_layer_size].v1()
                                    + deriv_input_gate
                                        * self_wgts[source_idx + target_idx * this_layer_size].v2()
                                    + deriv_output_gate
                                        * self_wgts[source_idx + target_idx * this_layer_size].v3()
                                    + deriv_forget_gate
                                        * self_wgts[source_idx + target_idx * this_layer_size].v4();
                                prev_step_this_layer_derivs[source_idx] += add_deriv;
                            };
                        }
                        let g_biases: &mut [F64x4] = &mut grad.layer_bias_parameters_mut(layer_idx);
                        g_biases[target_idx].add(deriv);
                    }
                    std::mem::swap(&mut self.state1, &mut self.state2);
                }
            }
        }
    }

    fn propagate_v2_shadow(
        &mut self,
        nn: &LSTMv2,
        inputs: &[f64],
        outputs: &mut Vec<f64>,
        collect_gradients: bool,
    ) {
        assert_eq!(inputs.len(), nn.layer_sizes[0]);
        assert_eq!(outputs.len(), nn.layer_sizes[nn.layer_sizes.len() - 1]);

        // if given, compute_gradients is a slice to the gradients of the output layer
        // with respect to the output of the network, and a mutable reference to gradients
        let mut backprop_step: Option<BackpropStep> = None;
        if collect_gradients {
            // TODO: make a scheme that can re-use backpropagation steps.
            let mut step = BackpropStep::new(&self, &nn.layer_sizes, inputs.len(), outputs.len());
            step.inputs.copy_from_slice(inputs);
            backprop_step = Some(step);
        }

        let mut state_offset: usize = 0;

        self.state1[0..inputs.len()].copy_from_slice(inputs);
        for layer_idx in 1..nn.layer_sizes.len() - 1 {
            let to_this_layer_wgts: &[F64x4] = nn.layer_to_layer_parameters(layer_idx);
            let self_wgts: &[F64x4] = nn.layer_to_self_parameters(layer_idx);
            let biases: &[F64x4] = nn.layer_bias_parameters(layer_idx);
            let this_layer_size: usize = nn.layer_sizes[layer_idx];
            let prev_layer_size: usize = nn.layer_sizes[layer_idx - 1];

            let state_offset_start = state_offset;
            let state_offset_end = state_offset + this_layer_size;

            let last_memories: &mut [f64] =
                &mut self.memories[state_offset..state_offset + this_layer_size];
            let last_activations: &mut [f64] =
                &mut self.last_activations[state_offset..state_offset + this_layer_size];

            state_offset += this_layer_size;

            for target_idx in 0..this_layer_size {
                // iiof = input, input gate, output gate, forget gate
                let mut iiof = biases[target_idx];
                // add activations from previous layer
                for source_idx in 0..prev_layer_size {
                    let input: f64 = self.state1[source_idx];
                    iiof.mul_add_scalar(
                        input,
                        to_this_layer_wgts[source_idx + target_idx * nn.layer_sizes[layer_idx - 1]],
                    );
                }
                // add self activations
                for source_idx in 0..this_layer_size {
                    let input: f64 = last_activations[source_idx];
                    iiof.mul_add_scalar(
                        input,
                        self_wgts[source_idx + target_idx * this_layer_size],
                    );
                }

                let mut iiof_sigmoid = iiof;
                iiof_sigmoid.fast_sigmoid();

                let input: f64 = iiof_sigmoid.v1() * 2.0 - 1.0;
                let input_gate_activation: f64 = iiof_sigmoid.v2();
                let output_gate_activation: f64 = iiof_sigmoid.v3();
                let forget_gate_activation: f64 = iiof_sigmoid.v4();

                let new_memory: f64 = input * input_gate_activation
                    + last_memories[target_idx] * forget_gate_activation;

                let new_activation: f64 = fast_tanh(new_memory) * output_gate_activation;

                self.state2[target_idx] = new_activation;
                // backpropagation tracking
                if let Some(ref mut step) = backprop_step {
                    let ga = step.gate_activations[state_offset_start..state_offset_end].as_mut();
                    ga[target_idx] = F64x4::new(
                        input,
                        input_gate_activation,
                        output_gate_activation,
                        forget_gate_activation,
                    );
                    let bp_last_memories =
                        step.last_memories[state_offset_start..state_offset_end].as_mut();
                    bp_last_memories[target_idx] = last_memories[target_idx];
                    let bp_new_memories =
                        step.memories[state_offset_start..state_offset_end].as_mut();
                    bp_new_memories[target_idx] = new_memory;
                }
                last_memories[target_idx] = new_memory;
            }
            last_activations.copy_from_slice(&self.state2[0..this_layer_size]);
            if let Some(ref mut step) = backprop_step {
                step.last_activations[state_offset_start..state_offset_end]
                    .copy_from_slice(&self.state2[0..this_layer_size]);
            }
            std::mem::swap(&mut self.state1, &mut self.state2);
        }

        // Output layer
        let output_wgts: &[f64] = nn.layer_to_output_parameters();
        let output_biases: &[f64] = nn.output_biases_parameters();
        let output_len: usize = nn.layer_sizes[nn.layer_sizes.len() - 1];
        let prev_layer_len: usize = nn.layer_sizes[nn.layer_sizes.len() - 2];
        for target_idx in 0..output_len {
            let mut output: f64 = 0.0;
            for source_idx in 0..prev_layer_len {
                output +=
                    self.state1[source_idx] * output_wgts[source_idx + target_idx * prev_layer_len];
            }
            output += output_biases[target_idx];
            outputs[target_idx] = fast_sigmoid(output);
            if collect_gradients {
                backprop_step
                    .as_mut()
                    .unwrap()
                    .output_prev_activations
                    .copy_from_slice(&self.state1[0..prev_layer_len]);
            }
        }

        if collect_gradients {
            backprop_step
                .as_mut()
                .unwrap()
                .last_outputs
                .copy_from_slice(outputs);
            self.backprop_steps
                .push(RefCell::new(backprop_step.unwrap()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    #[test]
    pub fn instantiation_succeeds() {
        let _nn = LSTMv2::new(&[10, 20]);
        let _nn2: LSTMv2 = LSTMv2::new(&[30, 40, 50, 60]);
    }

    #[test]
    pub fn no_overlapping_slices_ptr_check() {
        // There are a bunch of functions in impl LSTMv2 that return slices.
        // The slices are not supposed to overlap. This test checks that.
        let nn: LSTMv2 = LSTMv2::new(&[17, 19, 23, 29]);
        let nn2: LSTMv2 = LSTMv2::new(&[11, 13]);

        // Pointer to slice start, and the size, in bytes
        let mut collected_slices: Vec<(*const u8, usize, String)> = vec![];
        let f64x4_sz: usize = std::mem::size_of::<F64x4>();
        let f64_sz: usize = std::mem::size_of::<f64>();
        for i in 1..nn.layer_sizes.len() - 1 {
            collected_slices.push((
                nn.layer_to_layer_parameters(i).as_ptr() as *const u8,
                nn.layer_to_layer_parameters(i).len() * f64x4_sz,
                format!("layer_to_layer_parameters({})", i),
            ));
            collected_slices.push((
                nn.layer_to_self_parameters(i).as_ptr() as *const u8,
                nn.layer_to_self_parameters(i).len() * f64x4_sz,
                format!("layer_to_self_parameters({})", i),
            ));
            collected_slices.push((
                nn.layer_bias_parameters(i).as_ptr() as *const u8,
                nn.layer_bias_parameters(i).len() * f64x4_sz,
                format!("layer_bias_parameters({})", i),
            ));
        }
        collected_slices.push((
            nn.layer_to_output_parameters().as_ptr() as *const u8,
            nn.layer_to_output_parameters().len() * f64_sz,
            "layer_to_output_parameters".to_string(),
        ));
        collected_slices.push((
            nn.output_biases_parameters().as_ptr() as *const u8,
            nn.output_biases_parameters().len() * f64_sz,
            "output_biases_parameters".to_string(),
        ));

        collected_slices.push((
            nn2.layer_to_output_parameters().as_ptr() as *const u8,
            nn2.layer_to_output_parameters().len() * f64_sz,
            "layer_to_output_parameters".to_string(),
        ));
        collected_slices.push((
            nn2.output_biases_parameters().as_ptr() as *const u8,
            nn2.output_biases_parameters().len() * f64_sz,
            "output_biases_parameters".to_string(),
        ));

        for i1 in 0..collected_slices.len() {
            for i2 in (i1 + 1)..collected_slices.len() {
                let (ptr1, sz1, ref msg1) = collected_slices[i1];
                let (ptr2, sz2, ref msg2) = collected_slices[i2];
                assert!(
                    ptr1.wrapping_add(sz1) <= ptr2 || ptr2.wrapping_add(sz2) <= ptr1,
                    "Overlap between {} and {}",
                    msg1,
                    msg2
                );
            }
        }
    }

    #[test]
    fn propagate_big_network() {
        let nn: LSTMv2 = LSTMv2::new(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let mut nn_state = nn.start_v2();
        let mut output_vec = vec![0.0; 100];
        for _ in 0..1000 {
            nn_state.propagate_v2(
                &nn,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                &mut output_vec,
            );
        }
    }

    #[test]
    fn to_vec_and_from_vec_are_id() {
        let mut nn: LSTMv2 = LSTMv2::new(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let mut rng = thread_rng();
        for p4 in nn.parameters.iter_mut() {
            *p4 = F64x4::new(
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            );
        }
        let (vec, ctx) = nn.to_vec();
        let nn2 = LSTMv2::from_vec(&vec, &ctx);
        assert_eq!(nn, nn2);
    }

    /*
     * This is code I used to compare the results of the Rust LSTMv2 with a Haskell version
     * that has automatically computed gradients. I might use it in the future so I decided to
     * leave this commented out.
     *
     * I've left the Haskell file in hs/Main.hs in case it is needed once more.
    #[test]
    fn haskell_comparison() {
        let mut nn: LSTMv2 = LSTMv2::new(&[2, 2, 2, 2, 2]);
        let vec_src: Vec<f64> = vec![
           parameter vector goes here
        ];

        let (_vec, ctx) = nn.to_vec();
        nn = LSTMv2::from_vec(&vec_src, &ctx);

        /*
        let mut rng = thread_rng();
        for p4 in nn.parameters.iter_mut() {
            *p4 = F64x4::new(
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
            );
        }
        let (vec, _ctx) = nn.to_vec();
        */
        let mut st = nn.start_v2();
        let mut out = vec![1.0, 1.0];
        st.propagate_collect_gradients_v2(&nn, &[0.311, 0.422], &mut out);
        st.set_output_derivs(&nn, &[1.0, 1.0]);
        st.propagate_collect_gradients_v2(&nn, &[0.422, 0.311], &mut out);
        st.set_output_derivs(&nn, &[1.0, 1.0]);
        let mut grad = nn.zero_like();
        st.backpropagate(&nn, &mut grad);

        //let jsonified = serde_json::to_string(&vec).unwrap();
        //println!("{:?}", jsonified);
        println!("{:?}", out);
    }
    */
}
