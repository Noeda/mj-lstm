use crate::rnn::*;
use crate::simd_common::{fast_sigmoid, fast_tanh};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};

// CfC neural network (I like calling it nematode network because it comes from worm research)
//
// Based on this paper https://www.nature.com/articles/s42256-022-00556-7
// And studying code read from https://github.com/raminmh/CfC
//
// @article{hasani_closed-form_2022,
//	title = {Closed-form continuous-time neural networks},
//	journal = {Nature Machine Intelligence},
//	author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
//	issn = {2522-5839},
//	month = nov,
//	year = {2022},
//}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct NematodeLayer {
    ninputs: usize,
    nhiddens: usize,
    noutputs: usize,

    nbackbone_units: usize,
    nbackbone_layers: usize,
    backbone_weights: Vec<Vec<f64>>,
    backbone_biases: Vec<Vec<f64>>,

    ff1_weights: Vec<f64>,
    ff1_biases: Vec<f64>,
    ff2_weights: Vec<f64>,
    ff2_biases: Vec<f64>,
    time_a_weights: Vec<f64>,
    time_a_biases: Vec<f64>,
    time_b_weights: Vec<f64>,
    time_b_biases: Vec<f64>,

    out_weights: Vec<f64>,
    out_biases: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct NematodeLayerState {
    network: NematodeLayer,
    out: Vec<f64>,
    hstate: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct Nematode {
    ninputs: usize,
    nhiddens: usize,
    nbackbone_units: usize,
    nbackbone_layers: usize,
    noutputs: usize,

    layers: Vec<NematodeLayer>,
}

pub struct NematodeState {
    layer_states: Vec<NematodeLayerState>,
}

impl Nematode {
    pub fn new(
        ninputs: usize,
        nlayers: usize,
        nhiddens: usize,
        nbackbone_units: usize,
        nbackbone_layers: usize,
        noutputs: usize,
    ) -> Self {
        assert!(nhiddens > 0);

        let mut layers = Vec::new();
        for idx in 0..nlayers {
            let layer_inputs = if idx == 0 { ninputs } else { nhiddens };
            let layer_outputs = if idx == nlayers - 1 {
                noutputs
            } else {
                nhiddens
            };
            layers.push(NematodeLayer::new(
                layer_inputs,
                layer_outputs,
                nhiddens,
                nbackbone_units,
                nbackbone_layers,
            ));
        }

        Self {
            ninputs,
            nhiddens,
            nbackbone_units,
            nbackbone_layers,
            noutputs,
            layers,
        }
    }

    fn start(&self) -> NematodeState {
        let mut layer_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layer_states.push(layer.start());
        }
        NematodeState { layer_states }
    }
}

impl NematodeLayer {
    pub fn new(
        ninputs: usize,
        noutputs: usize,
        nhidden: usize,
        nbackbone_units: usize,
        nbackbone_layers: usize,
    ) -> Self {
        let mut backbone_weights = Vec::with_capacity(1 + nbackbone_layers);
        let mut backbone_biases = Vec::with_capacity(1 + nbackbone_layers);
        // inputs->hidden (always exists)
        backbone_weights.push(vec![0.0; (ninputs + nhidden) * nbackbone_units]);
        backbone_biases.push(vec![0.0; nbackbone_units]);
        // hidden->hidden (optional)
        for _ in 0..nbackbone_layers {
            backbone_weights.push(vec![0.0; nbackbone_units * nbackbone_units]);
            backbone_biases.push(vec![0.0; nbackbone_units]);
        }

        let ff1_weights = vec![0.0; nbackbone_units * nhidden];
        let ff1_biases = vec![0.0; nhidden];
        let ff2_weights = vec![0.0; nbackbone_units * nhidden];
        let ff2_biases = vec![0.0; nhidden];
        let time_a_weights = vec![0.0; nbackbone_units * nhidden];
        let time_a_biases = vec![0.0; nhidden];
        let time_b_weights = vec![0.0; nbackbone_units * nhidden];
        let time_b_biases = vec![0.0; nhidden];
        let out_weights = vec![0.0; nhidden * noutputs];
        let out_biases = vec![0.0; noutputs];

        Self {
            ninputs,
            noutputs,
            nhiddens: nhidden,
            nbackbone_units,
            nbackbone_layers,
            backbone_weights,
            backbone_biases,
            ff1_weights,
            ff1_biases,
            ff2_weights,
            ff2_biases,
            time_a_weights,
            time_a_biases,
            time_b_weights,
            time_b_biases,
            out_weights,
            out_biases,
        }
    }

    fn start(&self) -> NematodeLayerState {
        NematodeLayerState {
            network: self.clone(),
            hstate: vec![0.0; self.nhiddens],
            out: vec![0.0; self.noutputs],
        }
    }
}

impl NematodeState {
    fn propagate(&mut self, inputs: &[f64]) -> &[f64] {
        let mut outputs: &[f64];
        let mut inputs: Vec<f64> = inputs.to_vec();
        for layer_state in self.layer_states.iter_mut() {
            outputs = layer_state.propagate(&mut inputs);
            inputs.clear();
            inputs.extend_from_slice(outputs);
        }
        self.layer_states.last().unwrap().out.as_slice()
    }
}

impl NematodeLayerState {
    fn propagate(&mut self, inputs: &mut Vec<f64>) -> &[f64] {
        assert!(inputs.len() == self.network.ninputs);

        // Set up intermediate vectors
        let capacity = std::cmp::max(
            self.network.ninputs + self.network.nhiddens,
            self.network.nbackbone_units,
        );
        inputs.reserve(capacity - inputs.len());
        for i in 0..self.network.nhiddens {
            inputs.push(self.hstate[i]);
        }
        while inputs.len() < capacity {
            inputs.push(0.0);
        }

        let mut state1: Vec<f64> = vec![];
        std::mem::swap(&mut state1, inputs);
        let mut state2: Vec<f64> = vec![0.0; capacity];

        let mut ff1_out: Vec<f64> = vec![0.0; self.network.nhiddens];
        let mut ff2_out: Vec<f64> = vec![0.0; self.network.nhiddens];
        let mut time_a_out: Vec<f64> = vec![0.0; self.network.nhiddens];
        let mut time_b_out: Vec<f64> = vec![0.0; self.network.nhiddens];

        // Backbone

        // input->hidden
        for i in 0..self.network.nbackbone_units {
            let mut sum = self.network.backbone_biases[0][i];
            for j in 0..self.network.ninputs + self.network.nhiddens {
                sum += state1[j]
                    * self.network.backbone_weights[0]
                        [i * (self.network.ninputs + self.network.nhiddens) + j];
            }
            state2[i] = fast_tanh(sum);
        }
        std::mem::swap(&mut state1, &mut state2);
        // hidden layers
        for l in 1..self.network.nbackbone_layers {
            for i in 0..self.network.nbackbone_units {
                let mut sum = self.network.backbone_biases[l][i];
                for j in 0..self.network.nbackbone_units {
                    sum += state1[j]
                        * self.network.backbone_weights[l][i * self.network.nbackbone_units + j];
                }
                state2[i] = fast_tanh(sum);
            }
            std::mem::swap(&mut state1, &mut state2);
        }

        // backbone output is in state1

        // CfC part
        // nbackbone -> nhiddens (ff1)
        for i in 0..self.network.nhiddens {
            let mut sum = self.network.ff1_biases[i];
            for j in 0..self.network.nbackbone_units {
                sum += state1[j] * self.network.ff1_weights[j * self.network.nhiddens + i];
            }
            ff1_out[i] = fast_tanh(sum);
        }
        // nbackbone -> nhiddens (ff2)
        for i in 0..self.network.nhiddens {
            let mut sum = self.network.ff2_biases[i];
            for j in 0..self.network.nbackbone_units {
                sum += state1[j] * self.network.ff2_weights[j * self.network.nhiddens + i];
            }
            ff2_out[i] = fast_tanh(sum);
        }
        // nbackbone -> time_a_out
        for i in 0..self.network.nhiddens {
            let mut sum = self.network.time_a_biases[i];
            for j in 0..self.network.nbackbone_units {
                sum += state1[j] * self.network.time_a_weights[j * self.network.nhiddens + i];
            }
            time_a_out[i] = sum;
        }
        // nbackbone -> time_b_out
        for i in 0..self.network.nhiddens {
            let mut sum = self.network.time_b_biases[i];
            for j in 0..self.network.nbackbone_units {
                sum += state1[j] * self.network.time_b_weights[j * self.network.nhiddens + i];
            }
            time_b_out[i] = sum;
        }

        // interpolate times
        for i in 0..self.network.nhiddens {
            let time_a = time_a_out[i];
            let time_b = time_b_out[i];
            // ts (assumed to always be 1)
            let ts: f64 = 1.0;
            let interp = fast_sigmoid(time_a * ts + time_b);

            self.hstate[i] = ff1_out[i] * (1.0 - interp) + ff2_out[i] * interp;
        }

        // hidden state is now updated in the layer.
        // next, compute the final output
        for i in 0..self.network.noutputs {
            let mut sum = self.network.out_biases[i];
            for j in 0..self.network.nhiddens {
                sum += self.hstate[j] * self.network.out_weights[j * self.network.noutputs + i];
            }
            self.out[i] = fast_sigmoid(sum);
        }
        std::mem::swap(inputs, &mut state1);
        &self.out
    }
}

impl RNN for Nematode {
    type RNNState = NematodeState;

    fn start(&self) -> Self::RNNState {
        Nematode::start(self)
    }
}

impl RNNState for NematodeState {
    type InputType = f64;
    type OutputType = f64;

    fn propagate(&mut self, inputs: &[f64]) -> &[f64] {
        NematodeState::propagate(self, inputs)
    }

    fn propagate32(&mut self, _inputs: &[f32]) -> &[f32] {
        unimplemented!();
    }

    fn reset(&mut self) {
        for layer in self.layer_states.iter_mut() {
            *layer = layer.network.start();
        }
    }
}

impl Vectorizable for Nematode {
    type Context = (
        usize,
        usize,
        usize,
        usize,
        usize,
        Vec<(usize, <NematodeLayer as Vectorizable>::Context)>,
    );

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut result: Vec<f64> = vec![];
        let mut ctxs: Vec<(usize, <NematodeLayer as Vectorizable>::Context)> = vec![];
        for l in self.layers.iter() {
            let (vec, ctx) = l.to_vec();
            let vec_len = vec.len();
            result.extend(vec);
            ctxs.push((vec_len, ctx));
        }
        (
            result,
            (
                self.ninputs,
                self.nhiddens,
                self.noutputs,
                self.nbackbone_units,
                self.nbackbone_layers,
                ctxs,
            ),
        )
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut vec: &[f64] = vec;
        let mut layers: Vec<NematodeLayer> = vec![];
        for (vec_len, ctx) in ctx.5.iter() {
            layers.push(NematodeLayer::from_vec(&vec[..*vec_len], ctx));
            vec = &vec[*vec_len..];
        }
        assert!(vec.is_empty());

        Nematode {
            ninputs: ctx.0,
            nhiddens: ctx.1,
            noutputs: ctx.2,
            nbackbone_units: ctx.3,
            nbackbone_layers: ctx.4,
            layers,
        }
    }
}

impl Vectorizable for NematodeLayer {
    type Context = (usize, usize, usize, usize, usize);

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut result: Vec<f64> = vec![];
        for w in self.backbone_weights.iter() {
            result.extend_from_slice(w);
        }
        for w in self.backbone_biases.iter() {
            result.extend_from_slice(w);
        }
        result.extend_from_slice(&self.ff1_weights);
        result.extend_from_slice(&self.ff1_biases);
        result.extend_from_slice(&self.ff2_weights);
        result.extend_from_slice(&self.ff2_biases);
        result.extend_from_slice(&self.time_a_weights);
        result.extend_from_slice(&self.time_a_biases);
        result.extend_from_slice(&self.time_b_weights);
        result.extend_from_slice(&self.time_b_biases);
        result.extend_from_slice(&self.out_weights);
        result.extend_from_slice(&self.out_biases);
        (
            result,
            (
                self.nbackbone_layers,
                self.nbackbone_units,
                self.ninputs,
                self.nhiddens,
                self.noutputs,
            ),
        )
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut vec: &[f64] = vec;
        let original_len = vec.len();
        let nbackbone_layers = ctx.0;
        let nbackbone_units = ctx.1;
        let ninputs = ctx.2;
        let nhiddens = ctx.3;
        let noutputs = ctx.4;
        let mut backbone_weights = Vec::with_capacity(1 + nbackbone_layers);
        // input->hidden
        backbone_weights.push(vec[..(nhiddens + ninputs) * nbackbone_units].to_vec());
        vec = &vec[(nhiddens + ninputs) * nbackbone_units..];
        for _ in 0..nbackbone_layers {
            backbone_weights.push(vec[..nbackbone_units * nbackbone_units].to_vec());
            vec = &vec[nbackbone_units * nbackbone_units..];
        }
        let mut backbone_biases = Vec::with_capacity(1 + nbackbone_layers);
        backbone_biases.push(vec[..nbackbone_units].to_vec());
        vec = &vec[nbackbone_units..];
        for _ in 0..nbackbone_layers {
            backbone_biases.push(vec[..nbackbone_units].to_vec());
            vec = &vec[nbackbone_units..];
        }
        let ff1_weights = vec[..nbackbone_units * nhiddens].to_vec();
        vec = &vec[nbackbone_units * nhiddens..];
        let ff1_biases = vec[..nhiddens].to_vec();
        vec = &vec[nhiddens..];
        let ff2_weights = vec[..nbackbone_units * nhiddens].to_vec();
        vec = &vec[nbackbone_units * nhiddens..];
        let ff2_biases = vec[..nhiddens].to_vec();
        vec = &vec[nhiddens..];
        let time_a_weights = vec[..nbackbone_units * nhiddens].to_vec();
        vec = &vec[nbackbone_units * nhiddens..];
        let time_a_biases = vec[..nhiddens].to_vec();
        vec = &vec[nhiddens..];
        let time_b_weights = vec[..nbackbone_units * nhiddens].to_vec();
        vec = &vec[nbackbone_units * nhiddens..];
        let time_b_biases = vec[..nhiddens].to_vec();
        vec = &vec[nhiddens..];
        let out_weights = vec[..nhiddens * noutputs].to_vec();
        vec = &vec[nhiddens * noutputs..];
        let out_biases = vec[..noutputs].to_vec();
        vec = &vec[noutputs..];
        assert!(vec.is_empty());

        Self {
            backbone_weights,
            backbone_biases,
            ff1_weights,
            ff1_biases,
            ff2_weights,
            ff2_biases,
            time_a_weights,
            time_a_biases,
            time_b_weights,
            time_b_biases,
            out_weights,
            out_biases,

            nbackbone_layers: ctx.0,
            nbackbone_units: ctx.1,
            ninputs: ctx.2,
            nhiddens: ctx.3,
            noutputs: ctx.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_vec_from_vec_id() {
        let net = Nematode::new(2, 3, 4, 5, 6, 7);
        let (vec, ctx) = net.to_vec();
        let net2 = Nematode::from_vec(&vec, &ctx);
        assert_eq!(net, net2);
    }

    #[test]
    fn smoke_test_propagate_layer() {
        let layer = NematodeLayer::new(2, 3, 4, 5, 6);
        let mut st = layer.start();

        let mut inp: Vec<f64> = vec![1.0, 2.0];
        let out = st.propagate(&mut inp);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn smoke_test_propagate() {
        let net = Nematode::new(2, 3, 4, 5, 6, 7);
        let mut st = net.start();

        let mut inp: Vec<f64> = vec![1.0, 2.0];
        let out = st.propagate(&mut inp);
        assert_eq!(out.len(), 7);
    }
}
