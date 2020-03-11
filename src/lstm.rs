use crate::rnn::{RNNState, RNN};
/// This is a MxN LSTM-node network.
///
/// It uses forget gates but does not have peepholes.
use rand::{thread_rng, Rng};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::mem;

use core::arch::x86_64::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LSTMNetwork {
    pub(crate) weights: Vec<Vec<M256Unpack>>,
    pub(crate) last_state_weights: Vec<Vec<M256Unpack>>,
    pub(crate) input_gate_biases: Vec<Vec<f64>>,
    pub(crate) output_gate_biases: Vec<Vec<f64>>,
    pub(crate) forget_gate_biases: Vec<Vec<f64>>,
    pub(crate) input_biases: Vec<Vec<f64>>,
    pub(crate) initial_memories: Vec<Vec<f64>>,

    pub(crate) output_layer_biases: Vec<f64>,
    pub(crate) output_layer_weights: Vec<f64>,

    pub(crate) widest_layer_size: usize,
    pub(crate) ninputs: usize,
    pub(crate) noutputs: usize,

    pub(crate) output_is_sigmoid: bool,
    pub(crate) no_output_bias: bool,
}

#[cfg(test)]
impl quickcheck::Arbitrary for LSTMNetwork {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let nlayers: usize = std::cmp::max(2, usize::arbitrary(g) % 20);
        let mut layer_sizes: Vec<usize> = vec![];
        for _ in 0..nlayers {
            layer_sizes.push(std::cmp::max(1, usize::arbitrary(g) % 30));
        }

        LSTMNetwork::new(&layer_sizes)
    }
}

struct Consuming<'a> {
    cursor: usize,
    vec: &'a [f64],
}

impl<'a> Consuming<'a> {
    fn new(vec: &'a [f64]) -> Self {
        Consuming { cursor: 0, vec }
    }

    fn extend(&mut self, another_vec: &mut Vec<f64>, items: usize) {
        another_vec.extend(&self.vec[self.cursor..self.cursor + items]);
        self.cursor += items;
    }
}

impl Vectorizable for LSTMNetwork {
    type Context = (Vec<usize>, bool, bool);

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut v = vec![];
        for layer in self.weights.iter() {
            v.extend(M256Unpack::v1_vec(layer));
        }
        for layer in self.weights.iter() {
            v.extend(M256Unpack::v2_vec(layer));
        }
        for layer in self.weights.iter() {
            v.extend(M256Unpack::v3_vec(layer));
        }
        for layer in self.weights.iter() {
            v.extend(M256Unpack::v4_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(M256Unpack::v1_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(M256Unpack::v2_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(M256Unpack::v3_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(M256Unpack::v4_vec(layer));
        }
        for layer in self.input_gate_biases.iter() {
            v.extend(layer);
        }
        for layer in self.output_gate_biases.iter() {
            v.extend(layer);
        }
        for layer in self.forget_gate_biases.iter() {
            v.extend(layer);
        }
        for layer in self.input_biases.iter() {
            v.extend(layer);
        }
        for layer in self.initial_memories.iter() {
            v.extend(layer);
        }
        v.extend(&self.output_layer_biases);
        v.extend(&self.output_layer_weights);

        let mut sizes = Vec::with_capacity(self.initial_memories.len() + 2);
        sizes.push(self.ninputs);
        for i in self.initial_memories.iter() {
            sizes.push(i.len());
        }
        sizes.push(self.noutputs);

        (v, (sizes, self.output_is_sigmoid, self.no_output_bias))
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let (sizes, output_is_sigmoid, no_output_bias) = ctx;
        let mut input_weights: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut input_gates: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut output_gates: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut forget_gates: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut last_state_weights_input: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut last_state_weights_input_gate: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut last_state_weights_output_gate: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut last_state_weights_forget_gate: Vec<Vec<f64>> = Vec::with_capacity(sizes.len() - 2);
        let mut input_gate_biases = Vec::with_capacity(sizes.len() - 2);
        let mut output_gate_biases = Vec::with_capacity(sizes.len() - 2);
        let mut forget_gate_biases = Vec::with_capacity(sizes.len() - 2);
        let mut input_biases = Vec::with_capacity(sizes.len() - 2);
        let mut initial_memories = Vec::with_capacity(sizes.len() - 2);
        let mut output_layer_biases = Vec::with_capacity(sizes[sizes.len() - 1]);
        let mut output_layer_weights =
            Vec::with_capacity(sizes[sizes.len() - 1] * sizes[sizes.len() - 2]);

        let mut c = Consuming::new(vec);

        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i] * sizes[i - 1]);
            input_weights.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i] * sizes[i - 1]);
            input_gates.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i] * sizes[i - 1]);
            output_gates.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i] * sizes[i - 1]);
            forget_gates.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            last_state_weights_input.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            last_state_weights_input_gate.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            last_state_weights_output_gate.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            last_state_weights_forget_gate.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            input_gate_biases.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            output_gate_biases.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            forget_gate_biases.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            input_biases.push(iw);
        }
        for i in 1..sizes.len() - 1 {
            let mut iw = vec![];
            c.extend(&mut iw, sizes[i]);
            initial_memories.push(iw);
        }
        c.extend(&mut output_layer_biases, sizes[sizes.len() - 1]);
        c.extend(
            &mut output_layer_weights,
            sizes[sizes.len() - 1] * sizes[sizes.len() - 2],
        );

        assert_eq!(c.cursor, vec.len());

        let mut widest = 0;
        for i in 0..sizes.len() {
            widest = std::cmp::max(widest, sizes[i]);
        }

        LSTMNetwork {
            weights: M256Unpack::vecs_from_vecs64(
                &input_weights,
                &input_gates,
                &output_gates,
                &forget_gates,
            ),
            input_gate_biases,
            output_gate_biases,
            forget_gate_biases,
            input_biases,
            last_state_weights: M256Unpack::vecs_from_vecs64(
                &last_state_weights_input,
                &last_state_weights_input_gate,
                &last_state_weights_output_gate,
                &last_state_weights_forget_gate,
            ),
            initial_memories,
            output_layer_biases,
            output_layer_weights,
            ninputs: sizes[0],
            noutputs: sizes[sizes.len() - 1],
            widest_layer_size: widest,
            output_is_sigmoid: *output_is_sigmoid,
            no_output_bias: *no_output_bias,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LSTMState {
    network: LSTMNetwork,
    memories: Vec<Vec<f64>>,
    last_activations: Vec<Vec<f64>>,
    storage1: Vec<f64>,
    storage2: Vec<f64>,
}

pub(crate) fn make_random_vec(len: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(thread_rng().gen_range(-0.001, 0.001));
    }
    v
}

fn make_random_vec4(len: usize) -> Vec<M256Unpack> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(unsafe {
            M256Unpack::new(
                thread_rng().gen_range(-0.001, 0.001),
                thread_rng().gen_range(-0.001, 0.001),
                thread_rng().gen_range(-0.001, 0.001),
                thread_rng().gen_range(-0.001, 0.001),
            )
        });
    }
    v
}

impl LSTMNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        if layer_sizes.len() < 2 {
            panic!("Must have at least 2 layers (for input and output)");
        }

        let mut weights = Vec::with_capacity(layer_sizes.len() - 2);
        let mut output_layer_weights = Vec::with_capacity(
            layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2],
        );
        let mut widest_layer_size = 0;
        for i in 0..layer_sizes.len() - 2 {
            weights.push(make_random_vec4(layer_sizes[i] * layer_sizes[i + 1]));
            widest_layer_size = std::cmp::max(widest_layer_size, layer_sizes[i]);
        }
        widest_layer_size = std::cmp::max(widest_layer_size, layer_sizes[layer_sizes.len() - 1]);
        widest_layer_size = std::cmp::max(widest_layer_size, layer_sizes[layer_sizes.len() - 2]);

        for _ in 0..layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2] {
            output_layer_weights.push(thread_rng().gen_range(-1.0, 1.0));
        }

        let mut memories = Vec::with_capacity(layer_sizes.len() - 2);
        let mut input_biases = Vec::with_capacity(layer_sizes.len() - 2);
        let mut output_gate_biases = Vec::with_capacity(layer_sizes.len() - 2);
        let mut forget_gate_biases = Vec::with_capacity(layer_sizes.len() - 2);
        let mut input_gate_biases = Vec::with_capacity(layer_sizes.len() - 2);

        let mut output_layer_biases = Vec::with_capacity(layer_sizes[layer_sizes.len() - 1]);
        for _ in 0..layer_sizes[layer_sizes.len() - 1] {
            output_layer_biases.push(thread_rng().gen_range(-1.0, 1.0));
        }

        let mut last_state_weights = Vec::with_capacity(layer_sizes.len() - 2);
        for i in 1..layer_sizes.len() - 1 {
            memories.push(make_random_vec(layer_sizes[i]));
            input_gate_biases.push(make_random_vec(layer_sizes[i]));
            output_gate_biases.push(make_random_vec(layer_sizes[i]));
            let onevec = vec![0.9999; layer_sizes[i]];
            forget_gate_biases.push(onevec);
            input_biases.push(make_random_vec(layer_sizes[i]));
            last_state_weights.push(make_random_vec4(layer_sizes[i]));
        }

        LSTMNetwork {
            weights,
            input_gate_biases,
            forget_gate_biases,
            output_gate_biases,
            output_layer_biases,
            last_state_weights,
            input_biases,
            output_layer_weights,
            initial_memories: memories,
            widest_layer_size,
            ninputs: layer_sizes[0],
            noutputs: *layer_sizes.last().unwrap(),
            output_is_sigmoid: true,
            no_output_bias: false,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.initial_memories.iter().map(|vec| vec.len()).sum()
    }

    pub fn set_no_output_bias(&mut self, no_bias: bool) {
        self.no_output_bias = no_bias;
    }

    pub fn no_output_bias(&self) -> bool {
        self.no_output_bias
    }

    pub fn set_output_is_sigmoid(&mut self, is_sigmoid: bool) {
        self.output_is_sigmoid = is_sigmoid;
    }

    pub fn output_is_sigmoid(&mut self) -> bool {
        self.output_is_sigmoid
    }

    pub fn num_inputs(&self) -> usize {
        self.ninputs
    }

    pub fn num_outputs(&self) -> usize {
        self.noutputs
    }

    pub fn layers(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.initial_memories.len() + 2);
        result.push(self.ninputs);
        for memories in self.initial_memories.iter() {
            result.push(memories.len());
        }
        result.push(self.noutputs);
        result
    }
}

impl RNN for LSTMNetwork {
    type RNNState = LSTMState;

    fn start(&self) -> LSTMState {
        let mut la: Vec<Vec<f64>> = Vec::with_capacity(self.initial_memories.len());
        for i in 0..self.initial_memories.len() {
            la.push(vec![0.0; self.initial_memories[i].len()]);
        }
        LSTMState {
            network: self.clone(),
            memories: self.initial_memories.clone(),
            last_activations: la,
            storage1: vec![0.0; self.widest_layer_size],
            storage2: vec![0.0; self.widest_layer_size],
        }
    }
}

impl RNNState for LSTMState {
    fn propagate<'b>(&mut self, inputs: &'b [f64]) -> &[f64] {
        self.lstm_propagate(inputs)
    }
}

impl LSTMState {
    #[inline]
    pub fn lstm_propagate<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        unsafe { self.lstm_propagate2(inputs) }
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn lstm_propagate2<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        assert_eq!(
            inputs.len() * self.network.initial_memories[0].len(),
            self.network.weights[0].len()
        );

        let mut outputs1 = &mut self.storage1;
        let mut outputs2 = &mut self.storage2;

        outputs2[..inputs.len()].clone_from_slice(&inputs[..]);

        let mut num_inputs = inputs.len();
        for i in 0..self.network.weights.len() {
            for tgt_idx in 0..self.network.initial_memories.get_unchecked(i).len() {
                let mut iiof = M256Unpack::new(
                    *self
                        .network
                        .input_biases
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx),
                    *self
                        .network
                        .input_gate_biases
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx),
                    *self
                        .network
                        .output_gate_biases
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx),
                    *self
                        .network
                        .forget_gate_biases
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx),
                );

                let last_act = self
                    .last_activations
                    .get_unchecked(i)
                    .get_unchecked(tgt_idx);
                iiof.mul_add_scalar(
                    *last_act,
                    *self
                        .network
                        .last_state_weights
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx),
                );

                for src_idx in 0..num_inputs {
                    let offset = src_idx + tgt_idx * num_inputs;
                    iiof.mul_add_scalar(
                        *outputs2.get_unchecked(src_idx),
                        *self.network.weights.get_unchecked(i).get_unchecked(offset),
                    );
                }

                let mut iiof_sigmoid = iiof;
                iiof_sigmoid.fast_sigmoid();

                let input_gate_s = iiof_sigmoid.v2();
                let input_s = iiof_sigmoid.v1() * 2.0 - 1.0;

                let new_memory = self.memories.get_unchecked(i).get_unchecked(tgt_idx)
                    * iiof_sigmoid.v4()
                    + input_s * input_gate_s;

                let output_s = iiof_sigmoid.v3();
                let output_v = (fast_sigmoid(new_memory) * 2.0 - 1.0) * output_s;

                *outputs1.get_unchecked_mut(tgt_idx) = output_v;
                *self
                    .memories
                    .get_unchecked_mut(i)
                    .get_unchecked_mut(tgt_idx) = new_memory;
                *self
                    .last_activations
                    .get_unchecked_mut(i)
                    .get_unchecked_mut(tgt_idx) = output_v;
            }
            mem::swap(&mut outputs1, &mut outputs2);
            num_inputs = self.memories.get_unchecked(i).len();
        }
        for tgt_idx in 0..self.network.noutputs {
            let mut v: f64 = if self.network.no_output_bias {
                0.0
            } else {
                *self.network.output_layer_biases.get_unchecked(tgt_idx)
            };
            for src_idx in 0..num_inputs {
                v += outputs2.get_unchecked(src_idx)
                    * self
                        .network
                        .output_layer_weights
                        .get_unchecked(src_idx + tgt_idx * num_inputs);
            }
            if self.network.output_is_sigmoid {
                *outputs1.get_unchecked_mut(tgt_idx) = fast_sigmoid(v);
            } else {
                *outputs1.get_unchecked_mut(tgt_idx) = v;
            }
        }
        &outputs1[0..self.network.noutputs]
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn fast_sigmoid(x: f64) -> f64 {
    0.5 + (x / (1.0 + x.abs())) * 0.5
}

pub fn inv_sigmoid(x: f64) -> f64 {
    if x <= 0.0 {
        return -100_000.0;
    }
    if x >= 1.0 {
        return 100_000.0;
    }
    -(1.0 / x - 1.0).ln()
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) union M256Unpack {
    val: __m256d,
    vec: Vec4,
}

impl PartialEq for M256Unpack {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.vec.eq(&other.vec) }
    }
}

impl Serialize for M256Unpack {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unsafe { self.vec.serialize(serializer) }
    }
}

impl<'de> Deserialize<'de> for M256Unpack {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec4 = Vec4::deserialize(deserializer)?;
        Ok(M256Unpack { vec: vec4 })
    }
}

impl fmt::Debug for M256Unpack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            write!(
                f,
                "M256Unpack {{ v1: {}, v2: {}, v3: {}, v4: {} }}",
                self.vec.v1, self.vec.v2, self.vec.v3, self.vec.v4
            )
        }
    }
}

impl M256Unpack {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn new(x1: f64, x2: f64, x3: f64, x4: f64) -> Self {
        M256Unpack {
            vec: Vec4 {
                v1: x1,
                v2: x2,
                v3: x3,
                v4: x4,
            },
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn fast_sigmoid(&mut self) {
        let half = _mm256_broadcast_sd(&0.5);
        let one = _mm256_broadcast_sd(&1.0);
        let negzero = _mm256_broadcast_sd(&-0.0);
        let self_abs = _mm256_andnot_pd(negzero, self.val);
        let plus_one = _mm256_add_pd(one, self_abs);
        let xdivided = _mm256_div_pd(self.val, plus_one);
        self.val = _mm256_fmadd_pd(xdivided, half, half)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn mul_add_scalar(&mut self, other1: f64, other2: M256Unpack) {
        let broadcast_other1: __m256d = _mm256_broadcast_sd(&other1);
        self.val = _mm256_fmadd_pd(broadcast_other1, other2.val, self.val);
    }

    fn vec_from_vec64(x1: &[f64], x2: &[f64], x3: &[f64], x4: &[f64]) -> Vec<M256Unpack> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(unsafe { M256Unpack::new(x1[idx], x2[idx], x3[idx], x4[idx]) });
        }

        result
    }

    fn vecs_from_vecs64(
        x1: &[Vec<f64>],
        x2: &[Vec<f64>],
        x3: &[Vec<f64>],
        x4: &[Vec<f64>],
    ) -> Vec<Vec<M256Unpack>> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(M256Unpack::vec_from_vec64(
                &x1[idx], &x2[idx], &x3[idx], &x4[idx],
            ));
        }

        result
    }

    fn v1_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v1());
        }
        result
    }

    fn v2_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v2());
        }
        result
    }

    fn v3_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v3());
        }
        result
    }

    fn v4_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v4());
        }
        result
    }

    #[inline]
    fn v1(&self) -> f64 {
        unsafe { self.vec.v1 }
    }

    #[inline]
    fn v2(&self) -> f64 {
        unsafe { self.vec.v2 }
    }

    #[inline]
    fn v3(&self) -> f64 {
        unsafe { self.vec.v3 }
    }

    #[inline]
    fn v4(&self) -> f64 {
        unsafe { self.vec.v4 }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
struct Vec4 {
    v1: f64,
    v2: f64,
    v3: f64,
    v4: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck! {
        fn to_vec_from_vec_id(network: LSTMNetwork) -> bool {
            let (vec, ctx) = network.to_vec();
            let new_network = LSTMNetwork::from_vec(&vec, &ctx);
            new_network == network
        }
    }

    quickcheck! {
        fn m256d_fast_sigmoid_works(x1: f64, x2: f64, x3: f64, x4: f64) -> bool {
            unsafe {
                let mut v = M256Unpack::new(x1, x2, x3, x4);
                v.fast_sigmoid();
                fast_sigmoid(x1) == v.v1() &&
                fast_sigmoid(x2) == v.v2() &&
                fast_sigmoid(x3) == v.v3() &&
                fast_sigmoid(x4) == v.v4()
            }
        }
    }
}
