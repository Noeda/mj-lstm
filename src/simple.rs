use crate::rnn::*;
use crate::simd_common::*;
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};

/// Simple feed-forward network using sigmoid activation function
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct SimpleNN {
    pub(crate) weights: Vec<Vec<f64>>,
    pub(crate) biases: Vec<Vec<f64>>,

    pub(crate) ninputs: usize,
    pub(crate) noutputs: usize,
    pub(crate) widest_layer: usize,
}

impl Vectorizable for SimpleNN {
    type Context = Vec<usize>;

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut layer_sizes = Vec::with_capacity(self.weights.len() + 1);
        layer_sizes.push(self.ninputs);

        let mut values = 0;
        for w in self.weights.iter() {
            values += w.len();
        }
        for b in self.biases.iter() {
            values += b.len();
            layer_sizes.push(b.len());
        }
        let mut out: Vec<f64> = Vec::with_capacity(values);
        for w in self.weights.iter() {
            out.extend(w.iter());
        }
        for b in self.biases.iter() {
            out.extend(b.iter());
        }
        (out, layer_sizes)
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut nn = SimpleNN::new(ctx);
        let mut cursor = 0;
        for w in nn.weights.iter_mut() {
            for w_value in w.iter_mut() {
                *w_value = vec[cursor];
                cursor += 1;
            }
        }
        for b in nn.biases.iter_mut() {
            for b_value in b.iter_mut() {
                *b_value = vec[cursor];
                cursor += 1;
            }
        }
        assert!(cursor == vec.len());
        nn
    }
}

impl SimpleNN {
    pub fn new(layer_sizes: &[usize]) -> Self {
        if layer_sizes.len() < 2 {
            panic!("Must have at least 2 layers (for input and output)");
        }

        let ninputs = layer_sizes[0];
        let noutputs = layer_sizes[layer_sizes.len() - 1];

        let mut previous_layer_size = ninputs;
        let mut weights = Vec::with_capacity(layer_sizes.len() - 1);
        let mut biases = Vec::with_capacity(layer_sizes.len() - 1);
        let mut widest_layer = ninputs;

        for layer_size in layer_sizes[1..].iter() {
            let layer_size = *layer_size;
            widest_layer = std::cmp::max(widest_layer, layer_size);

            weights.push(vec![0.0; previous_layer_size * layer_size]);
            biases.push(vec![0.0; layer_size]);
            previous_layer_size = layer_size;
        }

        SimpleNN {
            ninputs,
            noutputs,
            weights,
            biases,
            widest_layer,
        }
    }

    pub fn layer_size(&self, layer: usize) -> usize {
        if layer == 0 {
            return self.ninputs;
        }
        return self.biases[layer - 1].len();
    }

    pub fn num_outputs(&self) -> usize {
        self.noutputs
    }

    pub fn propagate<'a>(
        &self,
        inputs: &[f64],
        outputs: &mut [f64],
        mut storage1: &'a mut [f64],
        mut storage2: &'a mut [f64],
    ) {
        if outputs.len() < self.noutputs {
            panic!("Output vector too small!");
        }
        if storage1.len() < self.widest_layer {
            panic!("Storage too small!");
        }
        if storage2.len() < self.widest_layer {
            panic!("Storage too small!");
        }

        for idx in 0..inputs.len() {
            storage1[idx] = inputs[idx];
        }

        for layer_idx in 0..self.weights.len() {
            let layer_idx = layer_idx + 1;

            for target_idx in 0..self.layer_size(layer_idx) {
                let mut sum = self.biases[layer_idx - 1][target_idx];
                for source_idx in 0..self.layer_size(layer_idx - 1) {
                    sum += self.weights[layer_idx - 1]
                        [source_idx + target_idx * self.layer_size(layer_idx - 1)]
                        * storage1[source_idx];
                }
                storage2[target_idx] = fast_sigmoid(sum);
            }
            std::mem::swap(&mut storage1, &mut storage2);
        }
        for idx in 0..outputs.len() {
            outputs[idx] = storage1[idx];
        }
    }
}

/// Simple RNN network.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct SimpleRNNBase {
    network: SimpleNN,
    nrecurrents: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct SimpleRNNState {
    network: SimpleRNNBase,
    memories: Vec<f64>,
    storage1: Vec<f64>,
    storage2: Vec<f64>,
    storage3: Vec<f64>,
    storage4: Vec<f32>,
}

impl Vectorizable for SimpleRNNBase {
    type Context = (Vec<usize>, usize);

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let (vec, ctx) = self.network.to_vec();
        (vec, (ctx, self.nrecurrents))
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        SimpleRNNBase {
            network: SimpleNN::from_vec(vec, &ctx.0),
            nrecurrents: ctx.1,
        }
    }
}

impl SimpleRNNBase {
    pub fn new(layer_sizes: &[usize], nrecurrents: usize) -> Self {
        let mut inner_layer_sizes: Vec<usize> = layer_sizes.to_vec();
        inner_layer_sizes[0] += nrecurrents;
        let ln = inner_layer_sizes.len();
        inner_layer_sizes[ln - 1] += nrecurrents;

        let nn = SimpleNN::new(&inner_layer_sizes);

        SimpleRNNBase {
            network: nn,
            nrecurrents,
        }
    }
}

impl RNN for SimpleRNNBase {
    type RNNState = SimpleRNNState;

    fn start(&self) -> Self::RNNState {
        SimpleRNNState {
            network: self.clone(),
            memories: vec![0.0; self.nrecurrents],
            storage1: vec![0.0; self.network.widest_layer],
            storage2: vec![0.0; self.network.widest_layer],
            storage3: vec![0.0; self.network.widest_layer],
            storage4: vec![0.0; self.network.widest_layer],
        }
    }
}

impl RNNState for SimpleRNNState {
    type InputType = f64;
    type OutputType = f64;

    fn reset(&mut self) {
        for m in self.memories.iter_mut() {
            *m = 0.0;
        }
    }

    fn propagate32<'a, 'b>(&'a mut self, inputs: &'b [f32]) -> &'a [f32] {
        let inputs64: Vec<f64> = inputs.iter().map(|x| *x as f64).collect();
        let outputs: Vec<f64> = self.propagate(&inputs64).iter().cloned().collect();
        for idx in 0..outputs.len() {
            self.storage4[idx] = outputs[idx] as f32;
        }
        &self.storage4
    }

    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [f64]) -> &'a [f64] {
        assert!(inputs.len() == self.network.network.ninputs - self.memories.len());
        for idx in 0..inputs.len() {
            self.storage3[idx] = inputs[idx];
        }
        for idx in 0..self.network.nrecurrents {
            self.storage3[inputs.len() + idx] = self.memories[idx];
        }
        self.network.network.propagate(
            inputs,
            &mut self.storage3,
            &mut self.storage1,
            &mut self.storage2,
        );
        let noutputs = self.network.network.num_outputs() - self.network.nrecurrents;
        for idx in 0..self.network.nrecurrents {
            self.memories[idx] = self.storage3[idx + noutputs];
        }
        &self.storage3[0..noutputs]
    }
}
