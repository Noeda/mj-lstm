use crate::rnn::{RNNState, RNN};
/// This is a MxN LSTM-node network.
///
/// It uses forget gates but does not have peepholes.
use rand::{thread_rng, Rng};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::mem;
use std::mem::MaybeUninit;

use core::arch::x86_64::*;

pub type LSTMNetwork = LSTMNetworkBase<f64, F64x4>;
pub type LSTMNetworkF32 = LSTMNetworkBase<f32, F32x8>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct LSTMNetworkBase<T, Unpack> {
    pub(crate) weights: Vec<Vec<Unpack>>,
    pub(crate) last_state_weights: Vec<Vec<Unpack>>,
    pub(crate) iiof_biases: Vec<Vec<Unpack>>,
    pub(crate) initial_memories: Vec<Vec<T>>,

    pub(crate) output_layer_biases: Vec<T>,
    pub(crate) output_layer_weights: Vec<T>,

    pub(crate) widest_layer_size: usize,
    pub(crate) ninputs: usize,
    pub(crate) noutputs: usize,

    pub(crate) output_is_sigmoid: bool,
    pub(crate) no_output_bias: bool,
}

pub trait FromF64 {
    fn from_f64(f: f64) -> Self;
}

impl FromF64 for f64 {
    #[inline]
    fn from_f64(f: f64) -> Self {
        f
    }
}

impl FromF64 for f32 {
    #[inline]
    fn from_f64(f: f64) -> Self {
        f as f32
    }
}

#[cfg(test)]
impl<
        T: 'static + Clone + Send + FromF64,
        Unpack: 'static + Clone + Send + Unpackable + std::fmt::Debug + AllocateWeights,
    > quickcheck::Arbitrary for LSTMNetworkBase<T, Unpack>
{
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let nlayers: usize = std::cmp::max(2, usize::arbitrary(g) % 20);
        let mut layer_sizes: Vec<usize> = vec![];
        for _ in 0..nlayers {
            layer_sizes.push(std::cmp::max(1, usize::arbitrary(g) % 30));
        }

        LSTMNetworkBase::new(&layer_sizes)
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

impl Vectorizable for LSTMNetworkF32 {
    type Context = (Vec<usize>, bool, bool);

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let f64lstm: LSTMNetwork = LSTMNetwork::from(self);
        f64lstm.to_vec()
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let f64lstm: LSTMNetwork = LSTMNetwork::from_vec(vec, ctx);
        LSTMNetworkF32::from(&f64lstm)
    }
}

impl Vectorizable for LSTMNetwork {
    type Context = (Vec<usize>, bool, bool);

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut v = vec![];
        for layer in self.weights.iter() {
            v.extend(F64x4::v1_vec(layer));
        }
        for layer in self.weights.iter() {
            v.extend(F64x4::v2_vec(layer));
        }
        for layer in self.weights.iter() {
            v.extend(F64x4::v3_vec(layer));
        }
        for layer in self.weights.iter() {
            v.extend(F64x4::v4_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(F64x4::v1_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(F64x4::v2_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(F64x4::v3_vec(layer));
        }
        for layer in self.last_state_weights.iter() {
            v.extend(F64x4::v4_vec(layer));
        }
        for layer in self.iiof_biases.iter() {
            v.extend(F64x4::v1_vec(layer));
        }
        for layer in self.iiof_biases.iter() {
            v.extend(F64x4::v2_vec(layer));
        }
        for layer in self.iiof_biases.iter() {
            v.extend(F64x4::v3_vec(layer));
        }
        for layer in self.iiof_biases.iter() {
            v.extend(F64x4::v4_vec(layer));
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
        let mut input_biases = Vec::with_capacity(sizes.len() - 2);
        let mut input_gate_biases = Vec::with_capacity(sizes.len() - 2);
        let mut output_gate_biases = Vec::with_capacity(sizes.len() - 2);
        let mut forget_gate_biases = Vec::with_capacity(sizes.len() - 2);
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
            input_biases.push(iw);
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
            weights: F64x4::vecs_from_vecs64(
                &input_weights,
                &input_gates,
                &output_gates,
                &forget_gates,
            ),
            iiof_biases: F64x4::vecs_from_vecs64(
                &input_biases,
                &input_gate_biases,
                &output_gate_biases,
                &forget_gate_biases,
            ),
            last_state_weights: F64x4::vecs_from_vecs64(
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

pub type LSTMState = LSTMStateBase<f64, F64x4>;
pub type LSTMStateF32 = LSTMStateBase<f32, F32x8>;

#[derive(Clone, Debug)]
pub struct LSTMStateBase<T, Unpack> {
    network: LSTMNetworkBase<T, Unpack>,
    memories: Vec<Vec<T>>,
    last_activations: Vec<Vec<T>>,
    storage1: Vec<T>,
    storage2: Vec<T>,
    storagef64: Vec<f64>,
}

pub(crate) fn make_random_vec<T: FromF64>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(T::from_f64(thread_rng().gen_range(-0.001, 0.001)));
    }
    v
}

fn make_random_vec4_pack_them_forgets_1<Unpack: Unpackable>(len: usize) -> Vec<Unpack> {
    let mut v = Vec::with_capacity(len);
    for i in 0..len {
        if i % 4 == 3 {
            v.push(thread_rng().gen_range(-0.999, 1.001));
        } else {
            v.push(thread_rng().gen_range(-0.001, 0.001));
        }
    }
    Unpack::from_f64_vec(&v)
}

fn make_random_vec4_pack_them<Unpack: Unpackable>(len: usize) -> Vec<Unpack> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(thread_rng().gen_range(-0.001, 0.001));
    }
    Unpack::from_f64_vec(&v)
}

/*
impl LSTMNetworkF32 {
    #[inline]
    pub fn weights_for(&self, layer: usize, src_idx: usize, tgt_idx: usize) -> &F32x8 {
        self.weights.get_unchecked(layer)
    }
}
*/

pub trait AllocateWeights {
    fn allocate_weights(layer_sizes_src: usize, layer_sizes_tgt: usize) -> Vec<Self>
    where
        Self: Sized;
}

impl AllocateWeights for F64x4 {
    fn allocate_weights(layer_sizes_src: usize, layer_sizes_tgt: usize) -> Vec<F64x4> {
        make_random_vec4_pack_them(layer_sizes_src * layer_sizes_tgt * 4)
    }
}

impl AllocateWeights for F32x8 {
    fn allocate_weights(layer_sizes_src: usize, layer_sizes_tgt: usize) -> Vec<F32x8> {
        if layer_sizes_tgt % 2 == 0 {
            make_random_vec4_pack_them(layer_sizes_src * layer_sizes_tgt * 4)
        } else {
            let mut vec: Vec<F32x8> =
                make_random_vec4_pack_them((layer_sizes_src * (layer_sizes_tgt + 1)) * 4);
            for src_idx in 0..layer_sizes_src {
                vec[src_idx + (layer_sizes_tgt / 2) * layer_sizes_src]
                    .vec
                    .v5 = 0.0;
                vec[src_idx + (layer_sizes_tgt / 2) * layer_sizes_src]
                    .vec
                    .v6 = 0.0;
                vec[src_idx + (layer_sizes_tgt / 2) * layer_sizes_src]
                    .vec
                    .v7 = 0.0;
                vec[src_idx + (layer_sizes_tgt / 2) * layer_sizes_src]
                    .vec
                    .v8 = 0.0;
            }
            vec
        }
    }
}

impl<T: 'static + Clone + FromF64, Unpack: Unpackable + std::fmt::Debug + AllocateWeights>
    LSTMNetworkBase<T, Unpack>
{
    pub fn new(layer_sizes: &[usize]) -> Self {
        if layer_sizes.len() < 2 {
            panic!("Must have at least 2 layers (for input and output)");
        }

        let mut weights = Vec::with_capacity(layer_sizes.len() - 2);
        let mut output_layer_weights: Vec<T> = Vec::with_capacity(
            layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2],
        );
        let mut widest_layer_size = 0;
        for i in 0..layer_sizes.len() - 2 {
            weights.push(Unpack::allocate_weights(layer_sizes[i], layer_sizes[i + 1]));
            widest_layer_size = std::cmp::max(widest_layer_size, layer_sizes[i]);
        }
        widest_layer_size = std::cmp::max(widest_layer_size, layer_sizes[layer_sizes.len() - 1]);
        widest_layer_size = std::cmp::max(widest_layer_size, layer_sizes[layer_sizes.len() - 2]);

        for _ in 0..layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2] {
            output_layer_weights.push(T::from_f64(thread_rng().gen_range(-1.0, 1.0)));
        }

        let mut memories = Vec::with_capacity(layer_sizes.len() - 2);
        let mut iiof_biases = Vec::with_capacity(layer_sizes.len() - 2);

        let mut output_layer_biases: Vec<T> =
            Vec::with_capacity(layer_sizes[layer_sizes.len() - 1]);
        for _ in 0..layer_sizes[layer_sizes.len() - 1] {
            output_layer_biases.push(T::from_f64(thread_rng().gen_range(-1.0, 1.0)));
        }

        let mut last_state_weights = Vec::with_capacity(layer_sizes.len() - 2);
        for i in 1..layer_sizes.len() - 1 {
            memories.push(make_random_vec(layer_sizes[i]));
            let iiof_vec = make_random_vec4_pack_them_forgets_1(layer_sizes[i] * 4);
            iiof_biases.push(iiof_vec);
            last_state_weights.push(make_random_vec4_pack_them(layer_sizes[i] * 4));
        }

        LSTMNetworkBase {
            weights,
            iiof_biases,
            output_layer_biases,
            last_state_weights,
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

impl<T: 'static + Clone + FromF64, Unpack: Clone> RNN for LSTMNetworkBase<T, Unpack> {
    type RNNState = LSTMStateBase<T, Unpack>;

    fn start(&self) -> LSTMStateBase<T, Unpack> {
        let mut la: Vec<Vec<T>> = Vec::with_capacity(self.initial_memories.len());
        for i in 0..self.initial_memories.len() {
            la.push(vec![T::from_f64(0.0); self.initial_memories[i].len()]);
        }
        LSTMStateBase {
            network: self.clone(),
            memories: self.initial_memories.clone(),
            last_activations: la,
            storage1: vec![T::from_f64(0.0); self.widest_layer_size],
            storage2: vec![T::from_f64(0.0); self.widest_layer_size],
            storagef64: vec![0.0; self.noutputs],
        }
    }
}

impl RNNState for LSTMStateBase<f64, F64x4> {
    fn propagate<'b>(&mut self, inputs: &'b [f64]) -> &[f64] {
        self.lstm_propagate(inputs)
    }
}

impl RNNState for LSTMStateBase<f32, F32x8> {
    fn propagate<'b>(&mut self, inputs: &'b [f64]) -> &[f64] {
        self.lstm_propagate(inputs)
    }
}

#[inline]
fn div2_round_up(x: usize) -> usize {
    x / 2 + x % 2
}

impl LSTMStateBase<f32, F32x8> {
    #[inline]
    pub fn lstm_propagate<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        unsafe { self.lstm_propagate2(inputs) }
    }

    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn lstm_propagate2<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        assert_eq!(inputs.len(), self.network.ninputs);

        let zero: f32 = 0.0;

        let mut outputs1: &mut [f32] = &mut self.storage1;
        let mut outputs2: &mut [f32] = &mut self.storage2;
        let storagef64: &mut [f64] = &mut self.storagef64;

        for i in 0..inputs.len() {
            *outputs2.get_unchecked_mut(i) = *inputs.get_unchecked(i) as f32;
        }

        let mut num_inputs = inputs.len();
        for i in 0..self.network.weights.len() {
            let layer_len = self.network.initial_memories.get_unchecked(i).len();
            for tgt_idx in 0..div2_round_up(layer_len) {
                let tgt_idx_m_2 = tgt_idx * 2;
                let tgt_idx_m_2_plus_1 = tgt_idx_m_2 + 1;
                let is_last = tgt_idx_m_2_plus_1 >= layer_len;
                let mut iiof = *self
                    .network
                    .iiof_biases
                    .get_unchecked(i)
                    .get_unchecked(tgt_idx);
                let last_act1 = self
                    .last_activations
                    .get_unchecked(i)
                    .get_unchecked(tgt_idx_m_2);
                let last_act2 = self
                    .last_activations
                    .get_unchecked(i)
                    .get(tgt_idx_m_2_plus_1)
                    .unwrap_or(&zero);
                iiof.mul_add_scalar2(
                    *last_act1,
                    *last_act2,
                    *self
                        .network
                        .last_state_weights
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx),
                );

                let tgt_idx_num_inputs = tgt_idx * num_inputs;
                for src_idx in 0..num_inputs {
                    let offset = src_idx + tgt_idx_num_inputs;
                    iiof.mul_add_scalar(
                        *outputs2.get_unchecked(src_idx),
                        *self.network.weights.get_unchecked(i).get_unchecked(offset),
                    );
                }

                let mut iiof_sigmoid = iiof;
                iiof_sigmoid.fast_sigmoid();

                let input_gate_s1 = iiof_sigmoid.v2();
                let input_s1 = iiof_sigmoid.v1() * 2.0 - 1.0;

                let new_memory1 = self.memories.get_unchecked(i).get_unchecked(tgt_idx_m_2)
                    * iiof_sigmoid.v4()
                    + input_s1 * input_gate_s1;

                let output_s1 = iiof_sigmoid.v3();
                let output_v1 = (fast_sigmoid32(new_memory1) * 2.0 - 1.0) * output_s1;

                *outputs1.get_unchecked_mut(tgt_idx_m_2) = output_v1;
                *self
                    .memories
                    .get_unchecked_mut(i)
                    .get_unchecked_mut(tgt_idx_m_2) = new_memory1;
                *self
                    .last_activations
                    .get_unchecked_mut(i)
                    .get_unchecked_mut(tgt_idx_m_2) = output_v1;

                if !is_last {
                    let input_gate_s2 = iiof_sigmoid.v6();
                    let input_s2 = iiof_sigmoid.v5() * 2.0 - 1.0;

                    let new_memory2 = self
                        .memories
                        .get_unchecked(i)
                        .get_unchecked(tgt_idx_m_2_plus_1)
                        * iiof_sigmoid.v8()
                        + input_s2 * input_gate_s2;

                    let output_s2 = iiof_sigmoid.v7();
                    let output_v2 = (fast_sigmoid32(new_memory2) * 2.0 - 1.0) * output_s2;

                    *outputs1.get_unchecked_mut(tgt_idx_m_2_plus_1) = output_v2;
                    *self
                        .memories
                        .get_unchecked_mut(i)
                        .get_unchecked_mut(tgt_idx_m_2_plus_1) = new_memory2;
                    *self
                        .last_activations
                        .get_unchecked_mut(i)
                        .get_unchecked_mut(tgt_idx_m_2_plus_1) = output_v2;
                }
            }
            mem::swap(&mut outputs1, &mut outputs2);
            num_inputs = layer_len;
        }
        for tgt_idx in 0..self.network.noutputs {
            let mut v: f32 = if self.network.no_output_bias {
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
                *outputs1.get_unchecked_mut(tgt_idx) = fast_sigmoid32(v);
            } else {
                *outputs1.get_unchecked_mut(tgt_idx) = v;
            }
        }
        for i in 0..self.network.noutputs {
            storagef64[i] = outputs1[i] as f64;
        }
        &storagef64[0..self.network.noutputs]
    }
}

impl LSTMStateBase<f64, F64x4> {
    #[inline]
    pub fn lstm_propagate<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        unsafe { self.lstm_propagate2(inputs) }
    }

    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn lstm_propagate2<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        if self.network.initial_memories.len() > 0 {
            assert_eq!(
                inputs.len() * self.network.initial_memories[0].len(),
                self.network.weights[0].len()
            );
        }

        let mut outputs1 = &mut self.storage1;
        let mut outputs2 = &mut self.storage2;

        outputs2[..inputs.len()].clone_from_slice(&inputs[..]);

        let mut num_inputs = inputs.len();
        for i in 0..self.network.weights.len() {
            let layer_size = self.network.initial_memories.get_unchecked(i).len();
            for tgt_idx in 0..layer_size {
                let mut iiof = *self
                    .network
                    .iiof_biases
                    .get_unchecked(i)
                    .get_unchecked(tgt_idx);
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
                    let item = self.network.weight_items(i, src_idx, tgt_idx);
                    iiof.mul_add_scalar(*outputs2.get_unchecked(src_idx), *item);
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

#[inline]
pub fn fast_sigmoid32(x: f32) -> f32 {
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
pub union F64x4 {
    val: __m256d,
    vec: Vec4_F64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union F32x8 {
    val: __m256,
    vec: Vec8_F32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union F32x4 {
    val: __m128,
    vec: Vec4_F32,
}

impl PartialOrd for F32x8 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        unsafe { self.vec.partial_cmp(&other.vec) }
    }
}

impl PartialOrd for F64x4 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        unsafe { self.vec.partial_cmp(&other.vec) }
    }
}

impl PartialEq for F32x8 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.vec.eq(&other.vec) }
    }
}

impl PartialEq for F64x4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.vec.eq(&other.vec) }
    }
}

impl Serialize for F64x4 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unsafe { self.vec.serialize(serializer) }
    }
}

impl<'de> Deserialize<'de> for F64x4 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec4 = Vec4_F64::deserialize(deserializer)?;
        Ok(F64x4 { vec: vec4 })
    }
}

impl fmt::Debug for F64x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            write!(
                f,
                "F64x4 {{ v1: {}, v2: {}, v3: {}, v4: {} }}",
                self.vec.v1, self.vec.v2, self.vec.v3, self.vec.v4
            )
        }
    }
}

impl fmt::Debug for F32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            write!(
                f,
                "F32x8 {{ v1: {}, v2: {}, v3: {}, v4: {}, v5: {}, v6: {}, v7: {}, v8: {} }}",
                self.vec.v1,
                self.vec.v2,
                self.vec.v3,
                self.vec.v4,
                self.vec.v5,
                self.vec.v6,
                self.vec.v7,
                self.vec.v8
            )
        }
    }
}

pub trait Unpackable {
    fn from_f64_vec(v: &[f64]) -> Vec<Self>
    where
        Self: Sized;
}

impl Unpackable for F64x4 {
    fn from_f64_vec(v: &[f64]) -> Vec<Self> {
        let mut result_len = v.len() / 4;
        if v.len() % 4 > 0 {
            result_len += 1;
        }
        let mut result = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let x1 = v.get(i * 4 + 0).unwrap_or(&0.0);
            let x2 = v.get(i * 4 + 1).unwrap_or(&0.0);
            let x3 = v.get(i * 4 + 2).unwrap_or(&0.0);
            let x4 = v.get(i * 4 + 3).unwrap_or(&0.0);

            result.push(unsafe { F64x4::new(*x1, *x2, *x3, *x4) });
        }
        result
    }
}

impl Unpackable for F32x8 {
    fn from_f64_vec(v: &[f64]) -> Vec<Self> {
        let mut result_len = v.len() / 8;
        if v.len() % 8 > 0 {
            result_len += 1;
        }
        let mut result = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let x1 = v.get(i * 8 + 0).unwrap_or(&0.0);
            let x2 = v.get(i * 8 + 1).unwrap_or(&0.0);
            let x3 = v.get(i * 8 + 2).unwrap_or(&0.0);
            let x4 = v.get(i * 8 + 3).unwrap_or(&0.0);
            let x5 = v.get(i * 8 + 4).unwrap_or(&0.0);
            let x6 = v.get(i * 8 + 5).unwrap_or(&0.0);
            let x7 = v.get(i * 8 + 6).unwrap_or(&0.0);
            let x8 = v.get(i * 8 + 7).unwrap_or(&0.0);

            result.push(unsafe {
                F32x8::new(
                    *x1 as f32, *x2 as f32, *x3 as f32, *x4 as f32, *x5 as f32, *x6 as f32,
                    *x7 as f32, *x8 as f32,
                )
            });
        }
        result
    }
}

fn vecf64_to_vecf32(v: &[F64x4]) -> Vec<F32x8> {
    let mut sz = v.len() / 2;
    if v.len() % 2 == 1 {
        sz += 1;
    }
    let mut result = Vec::with_capacity(sz);
    unsafe {
        let zero: F64x4 = F64x4::new(0.0, 0.0, 0.0, 0.0);
        for i in 0..sz {
            let p1 = v.get(i * 2).unwrap_or(&zero);
            let p2 = v.get(i * 2 + 1).unwrap_or(&zero);
            result.push(F32x8::new(
                p1.vec.v1 as f32,
                p1.vec.v2 as f32,
                p1.vec.v3 as f32,
                p1.vec.v4 as f32,
                p2.vec.v1 as f32,
                p2.vec.v2 as f32,
                p2.vec.v3 as f32,
                p2.vec.v4 as f32,
            ));
        }
    }
    result
}

fn vecf32_to_vecf64(v: &[F32x8]) -> Vec<F64x4> {
    let sz = v.len() * 2;
    let mut result = Vec::with_capacity(sz);
    for i in 0..v.len() {
        unsafe {
            result.push(F64x4::new(
                v[i].vec.v1 as f64,
                v[i].vec.v2 as f64,
                v[i].vec.v3 as f64,
                v[i].vec.v4 as f64,
            ));
            result.push(F64x4::new(
                v[i].vec.v5 as f64,
                v[i].vec.v6 as f64,
                v[i].vec.v7 as f64,
                v[i].vec.v8 as f64,
            ));
        }
    }
    result
}

impl From<&LSTMNetworkF32> for LSTMNetwork {
    fn from(other: &LSTMNetworkF32) -> Self {
        let nlayers = other.initial_memories.len() + 2;
        let layer_size = |n: usize| -> usize {
            if n == 0 {
                other.ninputs
            } else if n == nlayers - 1 {
                other.noutputs
            } else {
                other.initial_memories[n - 1].len()
            }
        };

        let mut weights: Vec<Vec<F64x4>> = Vec::with_capacity(nlayers - 2);
        for i in 0..nlayers - 2 {
            let desired_sz = layer_size(i) * layer_size(i + 1);
            let mut w_vec = unsafe { vec![F64x4::new(0.0, 0.0, 0.0, 0.0); desired_sz] };
            for tgt_idx in 0..layer_size(i + 1) {
                for src_idx in 0..layer_size(i) {
                    let src_wgt = unsafe { other.weight_items(i, src_idx, tgt_idx) };
                    let tgt_offset = src_idx + tgt_idx * layer_size(i);
                    w_vec[tgt_offset].vec.v1 = src_wgt.v1() as f64;
                    w_vec[tgt_offset].vec.v2 = src_wgt.v2() as f64;
                    w_vec[tgt_offset].vec.v3 = src_wgt.v3() as f64;
                    w_vec[tgt_offset].vec.v4 = src_wgt.v4() as f64;
                }
            }
            weights.push(w_vec);
        }
        assert_eq!(weights.len(), other.weights.len());
        let mut last_state_weights: Vec<Vec<F64x4>> = Vec::with_capacity(nlayers - 2);
        for i in 1..nlayers - 1 {
            let desired_sz = layer_size(i);
            let mut w_vec = vecf32_to_vecf64(&other.last_state_weights[i - 1]);
            if w_vec.len() != desired_sz {
                w_vec.truncate(desired_sz);
            }
            last_state_weights.push(w_vec);
        }
        assert_eq!(last_state_weights.len(), other.last_state_weights.len());

        let mut iiof_biases: Vec<Vec<F64x4>> = Vec::with_capacity(nlayers - 2);
        for i in 0..nlayers - 2 {
            let mut iiof = vecf32_to_vecf64(&other.iiof_biases[i]);
            if layer_size(i + 1) % 2 == 1 {
                iiof.truncate(iiof.len() - 1);
            }
            iiof_biases.push(iiof);
        }

        LSTMNetwork {
            weights,
            last_state_weights,
            iiof_biases,
            initial_memories: other
                .initial_memories
                .iter()
                .map(|inner| inner.iter().map(|x| *x as f64).collect())
                .collect(),
            output_layer_biases: other
                .output_layer_biases
                .iter()
                .map(|x| *x as f64)
                .collect(),
            output_layer_weights: other
                .output_layer_weights
                .iter()
                .map(|x| *x as f64)
                .collect(),
            widest_layer_size: other.widest_layer_size,
            ninputs: other.ninputs,
            noutputs: other.noutputs,
            output_is_sigmoid: other.output_is_sigmoid,
            no_output_bias: other.no_output_bias,
        }
    }
}

impl From<&LSTMNetwork> for LSTMNetworkF32 {
    fn from(other: &LSTMNetwork) -> Self {
        let nlayers = other.initial_memories.len() + 2;
        let layer_size = |n: usize| -> usize {
            if n == 0 {
                other.ninputs
            } else if n == nlayers - 1 {
                other.noutputs
            } else {
                other.initial_memories[n - 1].len()
            }
        };
        let mut weights: Vec<Vec<F32x8>> = Vec::with_capacity(nlayers - 2);
        for i in 0..nlayers - 2 {
            let desired_sz = if layer_size(i + 1) % 2 == 1 {
                div2_round_up(layer_size(i) * (layer_size(i + 1) + 1))
            } else {
                div2_round_up(layer_size(i) * layer_size(i + 1))
            };
            let mut w_vec =
                unsafe { vec![F32x8::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); desired_sz] };
            for tgt_idx in 0..layer_size(i + 1) {
                for src_idx in 0..layer_size(i) {
                    let src_wgt = other.weight_items(i, src_idx, tgt_idx);
                    let tgt_offset = src_idx + (tgt_idx / 2) * layer_size(i);
                    if tgt_idx % 2 == 0 {
                        w_vec[tgt_offset].vec.v1 = src_wgt.v1() as f32;
                        w_vec[tgt_offset].vec.v2 = src_wgt.v2() as f32;
                        w_vec[tgt_offset].vec.v3 = src_wgt.v3() as f32;
                        w_vec[tgt_offset].vec.v4 = src_wgt.v4() as f32;
                    } else {
                        w_vec[tgt_offset].vec.v5 = src_wgt.v1() as f32;
                        w_vec[tgt_offset].vec.v6 = src_wgt.v2() as f32;
                        w_vec[tgt_offset].vec.v7 = src_wgt.v3() as f32;
                        w_vec[tgt_offset].vec.v8 = src_wgt.v4() as f32;
                    }
                }
            }
            weights.push(w_vec);
        }

        let mut iiof_biases = Vec::with_capacity(nlayers - 2);
        for i in 0..nlayers - 2 {
            iiof_biases.push(vecf64_to_vecf32(&other.iiof_biases[i]));
        }

        LSTMNetworkF32 {
            weights,
            last_state_weights: other
                .last_state_weights
                .iter()
                .map(|inner| vecf64_to_vecf32(&inner))
                .collect(),
            iiof_biases,
            initial_memories: other
                .initial_memories
                .iter()
                .map(|inner| inner.iter().map(|x| *x as f32).collect())
                .collect(),
            output_layer_biases: other
                .output_layer_biases
                .iter()
                .map(|x| *x as f32)
                .collect(),
            output_layer_weights: other
                .output_layer_weights
                .iter()
                .map(|x| *x as f32)
                .collect(),
            widest_layer_size: other.widest_layer_size,
            ninputs: other.ninputs,
            noutputs: other.noutputs,
            output_is_sigmoid: other.output_is_sigmoid,
            no_output_bias: other.no_output_bias,
        }
    }
}

impl F32x4 {
    #[inline]
    fn v1(&self) -> f32 {
        unsafe { self.vec.v1 }
    }

    #[inline]
    fn v2(&self) -> f32 {
        unsafe { self.vec.v2 }
    }

    #[inline]
    fn v3(&self) -> f32 {
        unsafe { self.vec.v3 }
    }

    #[inline]
    fn v4(&self) -> f32 {
        unsafe { self.vec.v4 }
    }
}

impl F32x8 {
    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn new(x1: f32, x2: f32, x3: f32, x4: f32, x5: f32, x6: f32, x7: f32, x8: f32) -> Self {
        F32x8 {
            vec: Vec8_F32 {
                v1: x1,
                v2: x2,
                v3: x3,
                v4: x4,
                v5: x5,
                v6: x6,
                v7: x7,
                v8: x8,
            },
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn mul_add_scalar(&mut self, other1: f32, other2: F32x8) {
        let broadcast_other1: __m256 = _mm256_broadcast_ss(&other1);
        self.val = _mm256_fmadd_ps(broadcast_other1, other2.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn mul_add_scalar2(&mut self, other1: f32, other2: f32, other3: F32x8) {
        let b: *mut __m256 = MaybeUninit::<__m256>::uninit().as_mut_ptr();
        let b1: *mut __m128 = b as *mut __m128;
        let b2: *mut __m128 = b1.add(1);
        *b1 = _mm_broadcast_ss(&other1);
        *b2 = _mm_broadcast_ss(&other2);
        self.val = _mm256_fmadd_ps(*b, other3.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn fast_sigmoid(&mut self) {
        let half = _mm256_broadcast_ss(&0.5);
        let one = _mm256_broadcast_ss(&1.0);
        let negzero = _mm256_broadcast_ss(&-0.0);
        let self_abs = _mm256_andnot_ps(negzero, self.val);
        let plus_one = _mm256_add_ps(one, self_abs);
        let xdivided = _mm256_div_ps(self.val, plus_one);
        self.val = _mm256_fmadd_ps(xdivided, half, half)
    }

    #[inline]
    fn v1(&self) -> f32 {
        unsafe { self.vec.v1 }
    }

    #[inline]
    fn v2(&self) -> f32 {
        unsafe { self.vec.v2 }
    }

    #[inline]
    fn v3(&self) -> f32 {
        unsafe { self.vec.v3 }
    }

    #[inline]
    fn v4(&self) -> f32 {
        unsafe { self.vec.v4 }
    }

    #[inline]
    fn v5(&self) -> f32 {
        unsafe { self.vec.v5 }
    }

    #[inline]
    fn v6(&self) -> f32 {
        unsafe { self.vec.v6 }
    }

    #[inline]
    fn v7(&self) -> f32 {
        unsafe { self.vec.v7 }
    }

    #[inline]
    fn v8(&self) -> f32 {
        unsafe { self.vec.v8 }
    }
}

impl F64x4 {
    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn new(x1: f64, x2: f64, x3: f64, x4: f64) -> Self {
        F64x4 {
            vec: Vec4_F64 {
                v1: x1,
                v2: x2,
                v3: x3,
                v4: x4,
            },
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
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
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn mul_add_scalar(&mut self, other1: f64, other2: F64x4) {
        let broadcast_other1: __m256d = _mm256_broadcast_sd(&other1);
        self.val = _mm256_fmadd_pd(broadcast_other1, other2.val, self.val);
    }

    fn vec_from_vec64(x1: &[f64], x2: &[f64], x3: &[f64], x4: &[f64]) -> Vec<F64x4> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(unsafe { F64x4::new(x1[idx], x2[idx], x3[idx], x4[idx]) });
        }

        result
    }

    fn vecs_from_vecs64(
        x1: &[Vec<f64>],
        x2: &[Vec<f64>],
        x3: &[Vec<f64>],
        x4: &[Vec<f64>],
    ) -> Vec<Vec<F64x4>> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(F64x4::vec_from_vec64(
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
struct Vec4_F64 {
    v1: f64,
    v2: f64,
    v3: f64,
    v4: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
struct Vec8_F32 {
    v1: f32,
    v2: f32,
    v3: f32,
    v4: f32,
    v5: f32,
    v6: f32,
    v7: f32,
    v8: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
struct Vec4_F32 {
    v1: f32,
    v2: f32,
    v3: f32,
    v4: f32,
}

impl LSTMNetworkF32 {
    #[inline]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn weight_items(&self, layer: usize, src_idx: usize, tgt_idx: usize) -> &F32x4 {
        let num_inputs = if layer == 0 {
            self.ninputs
        } else {
            self.initial_memories.get_unchecked(layer - 1).len()
        };
        let f: &F32x8 = self
            .weights
            .get_unchecked(layer)
            .get_unchecked(src_idx + (tgt_idx / 2) * num_inputs);
        let f2: *const F32x8 = f;
        let f3: *const F32x4 = f2 as *const F32x4;
        let f4 = f3.add(tgt_idx % 2);
        let f5: &F32x4 = &*(f4 as *const F32x4);
        f5
    }
}

impl LSTMNetwork {
    #[inline]
    fn weight_items(&self, layer: usize, src_idx: usize, tgt_idx: usize) -> &F64x4 {
        unsafe {
            let num_inputs = if layer == 0 {
                self.ninputs
            } else {
                self.initial_memories.get_unchecked(layer - 1).len()
            };
            self.weights
                .get_unchecked(layer)
                .get_unchecked(src_idx + tgt_idx * num_inputs)
        }
    }
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
                let mut v = F64x4::new(x1, x2, x3, x4);
                v.fast_sigmoid();
                fast_sigmoid(x1) == v.v1() &&
                fast_sigmoid(x2) == v.v2() &&
                fast_sigmoid(x3) == v.v3() &&
                fast_sigmoid(x4) == v.v4()
            }
        }
    }

    quickcheck! {
        fn m256_fast_sigmoid_works(x1: f32, x2: f32, x3: f32, x4: f32, x5: f32, x6: f32, x7:f32, x8:f32) -> bool {
            unsafe {
                let mut v = F32x8::new(x1, x2, x3, x4, x5, x6, x7, x8);
                v.fast_sigmoid();
                fast_sigmoid32(x1) == v.v1() &&
                fast_sigmoid32(x2) == v.v2() &&
                fast_sigmoid32(x3) == v.v3() &&
                fast_sigmoid32(x4) == v.v4() &&
                fast_sigmoid32(x5) == v.v5() &&
                fast_sigmoid32(x6) == v.v6() &&
                fast_sigmoid32(x7) == v.v7() &&
                fast_sigmoid32(x8) == v.v8()
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar_works(x1: f64, v1: f64, v2: f64) -> bool {
            unsafe {
                let mut vec1 = F64x4::new(v1, v1 * 2.0, v1 * 3.0, v1 * 4.0);
                let original = vec1.clone();
                let vec2 = F64x4::new(v2, v2, v2, v2);
                vec1.mul_add_scalar(x1, vec2);
                (vec1.v1() - (vec2.v1()*x1 + original.v1())) < 0.001 &&
                (vec1.v2() - (vec2.v2()*x1 + original.v2())) < 0.001 &&
                (vec1.v3() - (vec2.v3()*x1 + original.v3())) < 0.001 &&
                (vec1.v4() - (vec2.v4()*x1 + original.v4())) < 0.001
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar2_f32_works(x1: f32, x2: f32, v1: f32, v2: f32) -> bool {
            unsafe {
                let mut vec1 = F32x8::new(v1, v1 * 2.0, v1 * 3.0, v1 * 4.0, v1 * 5.0, v1 * 6.0, v1 * 7.0, v1*8.0);
                let original = vec1.clone();
                let vec2 = F32x8::new(v2, v2 + 1.0, v2+2.0, v2+3.0, v2+4.0, v2+5.0, v2+6.0, v2+7.0);
                vec1.mul_add_scalar2(x1, x2, vec2);
                (vec1.v1() - (vec2.v1()*x1 + original.v1())) < 0.001 &&
                (vec1.v2() - (vec2.v2()*x1 + original.v2())) < 0.001 &&
                (vec1.v3() - (vec2.v3()*x1 + original.v3())) < 0.001 &&
                (vec1.v4() - (vec2.v4()*x1 + original.v4())) < 0.001 &&
                (vec1.v5() - (vec2.v5()*x2 + original.v5())) < 0.001 &&
                (vec1.v6() - (vec2.v6()*x2 + original.v6())) < 0.001 &&
                (vec1.v7() - (vec2.v7()*x2 + original.v7())) < 0.001 &&
                (vec1.v8() - (vec2.v8()*x2 + original.v8())) < 0.001
            }
        }
    }

    quickcheck! {
        fn from_f32_lstm_f64_lstm(net: LSTMNetworkF32) -> bool {
            let net32: LSTMNetwork = LSTMNetwork::from(&net);
            let net_back: LSTMNetworkF32 = LSTMNetworkF32::from(&net32);
            net_back.partial_cmp(&net) == Some(std::cmp::Ordering::Equal)
        }
    }

    quickcheck! {
        fn f64_network_smoke_test(net: LSTMNetwork) -> bool {
            let mut st = net.start();
            let input = vec![0.0; net.num_inputs()];
            st.propagate(&input);
            st.propagate(&input);
            st.propagate(&input);
            st.propagate(&input);
            st.propagate(&input);
            true
        }
    }

    quickcheck! {
        fn f32_and_f64_network_are_consistent(net: LSTMNetworkF32) -> bool {
            let net32: LSTMNetwork = LSTMNetwork::from(&net);
            let mut st1 = net.start();
            let mut st2 = net32.start();
            let mut input1: Vec<f32> = Vec::with_capacity(net.num_inputs());
            for i in 0..net.num_inputs() {
                input1.push(i as f32);
            }
            let input1_f64: Vec<f64> = input1.iter().map(|x| *x as f64).collect();

            let output1_f32: Vec<f64> = st1.propagate(&input1_f64).to_vec();
            let output1_f64: Vec<f64> = st2.propagate(&input1_f64).to_vec();

            let mut okay = true;
            for i in 0..output1_f32.len() {
                if (output1_f32[i] - output1_f64[i]).abs() > 0.1 {
                    okay = false;
                }
            }

            let output1_f32: Vec<f64> = st1.propagate(&input1_f64).to_vec();
            let output1_f64: Vec<f64> = st2.propagate(&input1_f64).to_vec();

            for i in 0..output1_f32.len() {
                if (output1_f32[i] - output1_f64[i]).abs() > 0.1 {
                    okay = false;
                }
            }

            okay
        }
    }
}
