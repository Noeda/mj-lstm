use crate::rnn::{RNNState, RNN};
use crate::simd_common::fast_sigmoid;
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub(crate) struct GRUW {
    z: f64,
    r: f64,
    h: f64,
}

impl GRUW {
    fn zero() -> Self {
        GRUW {
            z: 0.0,
            r: 0.0,
            h: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUNetwork {
    pub(crate) i_to_h_weights: Vec<Vec<GRUW>>,
    pub(crate) h_to_h_weights: Vec<Vec<GRUW>>,
    pub(crate) biases: Vec<Vec<GRUW>>,
    pub(crate) output_weights: Vec<f64>,
    pub(crate) output_biases: Vec<f64>,
    pub(crate) layer_sizes: Vec<usize>,
    pub(crate) widest_layer: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUState {
    gru: GRUNetwork,
    memories: Vec<Vec<f64>>,
    memories2: Vec<Vec<f64>>,
    outputs: Vec<f64>,
    storage1: Vec<f64>,
    storage2: Vec<f64>,
}

impl Vectorizable for GRUNetwork {
    type Context = Vec<usize>;

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut out = vec![];
        for w in self.i_to_h_weights.iter() {
            for gruw in w.iter() {
                out.push(gruw.z);
                out.push(gruw.r);
                out.push(gruw.h);
            }
        }
        for w in self.h_to_h_weights.iter() {
            for gruw in w.iter() {
                out.push(gruw.z);
                out.push(gruw.r);
                out.push(gruw.h);
            }
        }
        for b in self.biases.iter() {
            for gruw in b.iter() {
                out.push(gruw.z);
                out.push(gruw.r);
                out.push(gruw.h);
            }
        }
        for w in self.output_weights.iter() {
            out.push(*w);
        }
        for b in self.output_biases.iter() {
            out.push(*b);
        }
        (out, self.layer_sizes.clone())
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut cursor = 0;
        let mut network = GRUNetwork::new(ctx);
        for w in network.i_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.z = vec[cursor];
                gruw.r = vec[cursor + 1];
                gruw.h = vec[cursor + 2];
                cursor += 3;
            }
        }
        for w in network.h_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.z = vec[cursor];
                gruw.r = vec[cursor + 1];
                gruw.h = vec[cursor + 2];
                cursor += 3;
            }
        }
        for b in network.biases.iter_mut() {
            for gruw in b.iter_mut() {
                gruw.z = vec[cursor];
                gruw.r = vec[cursor + 1];
                gruw.h = vec[cursor + 2];
                cursor += 3;
            }
        }
        for w in network.output_weights.iter_mut() {
            *w = vec[cursor];
            cursor += 1;
        }
        for b in network.output_biases.iter_mut() {
            *b = vec[cursor];
            cursor += 1;
        }
        assert!(cursor == vec.len());
        network
    }
}

impl GRUNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        if layer_sizes.len() < 2 {
            panic!("Must have at least 2 layers (for input and output)");
        }

        let mut i_to_h_weights = Vec::with_capacity(layer_sizes.len() - 1);
        let mut h_to_h_weights = Vec::with_capacity(layer_sizes.len() - 1);
        let mut biases = Vec::with_capacity(layer_sizes.len() - 1);

        let mut widest_layer = layer_sizes[0];

        for (idx, layer) in layer_sizes[1..layer_sizes.len() - 1].iter().enumerate() {
            let prev_layer = layer_sizes[idx];
            let layer = *layer;
            i_to_h_weights.push(vec![GRUW::zero(); prev_layer * layer]);
            h_to_h_weights.push(vec![GRUW::zero(); layer * layer]);
            biases.push(vec![GRUW::zero(); layer]);
            widest_layer = std::cmp::max(widest_layer, layer);
        }

        let output_weights =
            vec![0.0; layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2]];
        let output_biases = vec![0.0; *layer_sizes.last().unwrap()];

        GRUNetwork {
            i_to_h_weights,
            h_to_h_weights,
            biases,
            output_weights,
            output_biases,
            layer_sizes: layer_sizes.to_vec(),
            widest_layer,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.layer_sizes[0]
    }

    pub fn num_outputs(&self) -> usize {
        *self.layer_sizes.last().unwrap()
    }

    pub fn layer_size(&self, layer: usize) -> usize {
        self.layer_sizes[layer]
    }
}

impl RNN for GRUNetwork {
    type RNNState = GRUState;

    fn start(&self) -> Self::RNNState {
        let mut memories: Vec<Vec<f64>> = Vec::with_capacity(self.layer_sizes.len() - 2);
        for b in self.biases.iter() {
            memories.push(vec![0.0; b.len()]);
        }

        let memories2 = memories.clone();

        GRUState {
            gru: self.clone(),
            memories,
            memories2,
            outputs: vec![],
            storage1: vec![],
            storage2: vec![],
        }
    }
}

impl RNNState for GRUState {
    fn reset(&mut self) {
        for v in self.memories.iter_mut() {
            for v2 in v.iter_mut() {
                *v2 = 0.0;
            }
        }
    }

    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [f64]) -> &'a [f64] {
        self.prop64(inputs);
        &self.outputs
    }

    fn propagate32<'a, 'b>(&'a mut self, _inputs: &'b [f32]) -> &'a [f32] {
        unimplemented!();
    }
}

impl GRUState {
    fn prop64<'a>(&mut self, inputs: &[f64]) {
        self.outputs.resize(self.gru.num_outputs(), 0.0);
        self.storage1.resize(self.gru.widest_layer, 0.0);
        self.storage2.resize(self.gru.widest_layer, 0.0);

        for idx in 0..inputs.len() {
            self.storage1[idx] = inputs[idx];
        }

        for layer_idx in 0..self.gru.biases.len() {
            let previous_layer_size = self.gru.layer_sizes[layer_idx];
            let layer_size = self.gru.layer_sizes[layer_idx + 1];
            for target_idx in 0..layer_size {
                // compute z[t]
                let mut zt = self.gru.biases[layer_idx][target_idx].z;
                for source_idx in 0..previous_layer_size {
                    zt += self.gru.i_to_h_weights[layer_idx]
                        [source_idx + target_idx * previous_layer_size]
                        .z
                        * self.storage1[source_idx];
                }
                for source_idx in 0..layer_size {
                    zt += self.gru.h_to_h_weights[layer_idx][source_idx + target_idx * layer_size]
                        .z
                        * self.memories[layer_idx][source_idx];
                }
                zt = fast_sigmoid(zt);

                // compute r[t]
                let mut rt = self.gru.biases[layer_idx][target_idx].r;
                for source_idx in 0..previous_layer_size {
                    rt += self.gru.i_to_h_weights[layer_idx]
                        [source_idx + target_idx * previous_layer_size]
                        .r
                        * self.storage1[source_idx];
                }
                for source_idx in 0..layer_size {
                    rt += self.gru.h_to_h_weights[layer_idx][source_idx + target_idx * layer_size]
                        .r
                        * self.memories[layer_idx][source_idx];
                }
                rt = fast_sigmoid(rt);

                // compute h^[t]
                let mut ht = self.gru.biases[layer_idx][target_idx].h;
                for source_idx in 0..previous_layer_size {
                    ht += self.gru.i_to_h_weights[layer_idx]
                        [source_idx + target_idx * previous_layer_size]
                        .h
                        * self.storage1[source_idx];
                }
                for source_idx in 0..layer_size {
                    ht += self.gru.h_to_h_weights[layer_idx][source_idx + target_idx * layer_size]
                        .h
                        * self.memories[layer_idx][source_idx]
                        * rt;
                }
                ht = fast_sigmoid(ht) * 2.0 - 1.0;

                // compute h[t] (next hidden state, also output)
                let ht_final = (1.0 - zt) * self.memories[layer_idx][target_idx] + zt * ht;
                self.memories2[layer_idx][target_idx] = ht_final;
                self.storage2[target_idx] = ht_final;
            }
            std::mem::swap(&mut self.storage1, &mut self.storage2);
        }
        for target_idx in 0..self.gru.num_outputs() {
            let mut sum = self.gru.output_biases[target_idx];
            let sz = self.gru.layer_sizes[self.gru.layer_sizes.len() - 2];
            for source_idx in 0..sz {
                sum += self.gru.output_weights[source_idx + target_idx * sz]
                    * self.storage1[source_idx];
            }
            sum = fast_sigmoid(sum);
            self.outputs[target_idx] = sum;
        }
        std::mem::swap(&mut self.memories, &mut self.memories2);
    }
}
