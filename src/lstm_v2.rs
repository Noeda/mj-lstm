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
//   * Biases on each LSTM layer and output layer
//   * Initial memory cell value for each LSTM

use crate::rnn::{RNNState, RNN};
use crate::simd_common::{fast_sigmoid, F64x4};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};

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
    last_activations: Vec<f64>,
    state1: Vec<f64>,
    state2: Vec<f64>,
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

fn layer_memory_cell_nparameters(num_hiddens: usize) -> usize {
    round_to_4(num_hiddens)
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
        // Initial states of LSTM cells
        num_parameters += layer_memory_cell_nparameters(layer_sizes[layer_idx]);
    }
    num_parameters
}

impl LSTMv2 {
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

    pub fn num_inputs(&self) -> usize {
        self.layer_sizes[0]
    }

    pub fn num_outputs(&self) -> usize {
        *self.layer_sizes.last().unwrap()
    }

    pub fn start_v2(&self) -> LSTMv2State {
        let mut memories: Vec<f64> = Vec::with_capacity(self.count_memory_cells());
        for i in 1..self.layer_sizes.len() - 1 {
            let initial_memory: &[f64] = self.layer_initial_memory_parameters(i);
            memories.extend_from_slice(initial_memory);
        }
        let memories_len = memories.len();
        LSTMv2State {
            memories,
            last_activations: vec![0.0; memories_len],
            state1: vec![0.0; self.widest_layer()],
            state2: vec![0.0; self.widest_layer()],
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

    /// Gives a slice parameters from previous layer to given layer.
    /// 0 is the input layer. 1 is considered to be the first LSTM layer.
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

    fn layer_initial_memory_parameters(&self, layer_idx: usize) -> &[f64] {
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
        // Skip over biases
        cursor += layer_biases_nparameters(self.layer_sizes[layer_idx]) / 4;
        let result: &[F64x4] = &self.parameters
            [cursor..cursor + layer_memory_cell_nparameters(self.layer_sizes[layer_idx]) / 4];
        // Return a slice with the proper size, without padding to a number divisible by 4
        unsafe {
            std::slice::from_raw_parts(result.as_ptr() as *const f64, self.layer_sizes[layer_idx])
        }
    }

    /// Slice to last LSTM layer (or input layer if there are no LSTM layers) to output layer parameters
    fn layer_to_output_parameters(&self) -> &[f64] {
        let cursor: usize = count_parameters(&self.layer_sizes, 1, self.layer_sizes.len() - 1) / 4;
        let result: &[F64x4] = &self.parameters[cursor
            ..cursor
                + layer_to_output_nparameters(
                    self.layer_sizes[self.layer_sizes.len() - 2],
                    self.layer_sizes[self.layer_sizes.len() - 1],
                ) / 4];
        // Return a slice with the proper size, without padding to a number divisible by 4
        unsafe {
            std::slice::from_raw_parts(
                result.as_ptr() as *const f64,
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
}

impl LSTMv2State {
    pub fn propagate_v2(&mut self, nn: &LSTMv2, inputs: &[f64], outputs: &mut Vec<f64>) {
        self.propagate_v2_shadow(nn, inputs, outputs, false);
    }

    fn propagate_v2_shadow(
        &mut self,
        nn: &LSTMv2,
        inputs: &[f64],
        outputs: &mut Vec<f64>,
        compute_gradients: bool,
    ) {
        assert_eq!(inputs.len(), nn.layer_sizes[0]);
        assert_eq!(outputs.len(), nn.layer_sizes[nn.layer_sizes.len() - 1]);

        let mut state_offset: usize = 0;

        self.state1[0..inputs.len()].copy_from_slice(inputs);
        for layer_idx in 1..nn.layer_sizes.len() - 1 {
            let to_this_layer_wgts: &[F64x4] = nn.layer_to_layer_parameters(layer_idx);
            let self_wgts: &[F64x4] = nn.layer_to_self_parameters(layer_idx);
            let biases: &[F64x4] = nn.layer_bias_parameters(layer_idx);
            let this_layer_size: usize = nn.layer_sizes[layer_idx];
            let prev_layer_size: usize = nn.layer_sizes[layer_idx - 1];
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

                let new_activation: f64 =
                    fast_sigmoid(new_memory * 2.0 - 1.0) * output_gate_activation;

                self.state2[target_idx] = new_activation;
                last_memories[target_idx] = new_memory;
            }
            last_activations.copy_from_slice(&self.state2[0..this_layer_size]);
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
            collected_slices.push((
                nn.layer_initial_memory_parameters(i).as_ptr() as *const u8,
                nn.layer_initial_memory_parameters(i).len() * f64_sz,
                format!("layer_initial_memory_parameters({})", i),
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
}
