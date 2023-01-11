use crate::rnn::{RNNState, RNN};
#[cfg(target_arch = "aarch64")]
use crate::simd_aarch64::*;
#[cfg(target_arch = "x86_64")]
use crate::simd_amd64::*;
use crate::simd_common::*;
use crate::unpackable::*;

/// This is a MxN LSTM-node network.
///
/// It uses forget gates but does not have peepholes.
use rand::{thread_rng, Rng};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::mem;

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

pub trait ToCStruct {
    /*
     * Converts the network into this C structure:
     *
     * struct {
     *   uint32_t nlayers;             // at least 2 (input and output)
     *   uint32_t output_is_sigmoid;   // 0 or 1
     *   uint32_t no_output_bias;      // 0 or 1
     *   uint32_t weights_offset;
     *   uint32_t biases_offset;
     *   uint32_t output_weights_offset;
     *   uint32_t output_biases_offset;
     *   uint32_t initial_memories_offset;
     *   uint32_t last_state_weights_offset;
     *   uint32_t[] layersizes;
     *   T[] values;
     * }
     *
     * The length of the structure will depend on the size of the network.
     */
    fn to_c_struct(&self, target: &mut Vec<u8>);
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
        Unpack: 'static + Clone + Send + Unpackable + AllocateWeights,
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

impl ToCStruct for LSTMNetwork {
    fn to_c_struct(&self, target: &mut Vec<u8>) {
        #[repr(C)]
        struct Header {
            nlayers: u32,
            output_is_sigmoid: u32,
            no_output_bias: u32,
            weights_offset: u32,
            biases_offset: u32,
            output_weights_offset: u32,
            output_biases_offset: u32,
            initial_memories_offset: u32,
            last_state_weights_offset: u32,
        }

        let mut header = Header {
            nlayers: 2 + self.weights.len() as u32,
            output_is_sigmoid: if self.output_is_sigmoid { 1 } else { 0 },
            no_output_bias: if self.no_output_bias { 1 } else { 0 },
            weights_offset: 0,
            biases_offset: 0,
            output_weights_offset: 0,
            output_biases_offset: 0,
            initial_memories_offset: 0,
            last_state_weights_offset: 0,
        };

        let mut layer_sizes: Vec<u32> = Vec::with_capacity(2 + self.weights.len());
        layer_sizes.push(self.ninputs as u32);
        for layer in self.last_state_weights.iter() {
            layer_sizes.push(layer.len() as u32);
        }
        layer_sizes.push(self.noutputs as u32);
        assert_eq!(layer_sizes.len(), header.nlayers as usize);

        let mut values: Vec<f64> = vec![];

        header.weights_offset = values.len() as u32;
        for layer in self.weights.iter() {
            for weight in layer.iter() {
                values.push(weight.v1());
                values.push(weight.v2());
                values.push(weight.v3());
                values.push(weight.v4());
            }
        }
        header.biases_offset = values.len() as u32;
        for layer in self.iiof_biases.iter() {
            for bias in layer.iter() {
                values.push(bias.v1());
                values.push(bias.v2());
                values.push(bias.v3());
                values.push(bias.v4());
            }
        }

        header.output_weights_offset = values.len() as u32;
        for weight in self.output_layer_weights.iter() {
            values.push(*weight);
        }

        header.output_biases_offset = values.len() as u32;
        for bias in self.output_layer_biases.iter() {
            values.push(*bias);
        }

        header.initial_memories_offset = values.len() as u32;
        for memory_layer in self.initial_memories.iter() {
            for memory in memory_layer.iter() {
                values.push(*memory);
            }
        }

        header.last_state_weights_offset = values.len() as u32;
        for last_state_layer in self.last_state_weights.iter() {
            for weight in last_state_layer.iter() {
                values.push(weight.v1());
                values.push(weight.v2());
                values.push(weight.v3());
                values.push(weight.v4());
            }
        }
        let values: Vec<f32> = values.into_iter().map(|x| x as f32).collect();

        target.truncate(0);
        // Update this whenever any field changes
        let header_sz = std::mem::size_of::<Header>();
        assert_eq!(header_sz, 9 * 4);
        target.reserve(values.len() * 4 + layer_sizes.len() * 4 + header_sz);

        unsafe {
            let target_ptr = target.as_mut_ptr();
            std::ptr::copy_nonoverlapping(&header as *const Header, target_ptr as *mut Header, 1);
            let target_ptr = target_ptr.byte_add(header_sz);
            std::ptr::copy_nonoverlapping(
                layer_sizes.as_ptr() as *const u32,
                target_ptr as *mut u32,
                layer_sizes.len(),
            );
            let target_ptr = target_ptr.byte_add(layer_sizes.len() * 4);
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const f32,
                target_ptr as *mut f32,
                values.len(),
            );
            target.set_len(values.len() * 4 + layer_sizes.len() * 4 + header_sz);
        }
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
    storagef32: Vec<f32>,
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
            v.push(thread_rng().gen_range(-2.0, 2.0));
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

impl<T: 'static + Clone + FromF64, Unpack: Unpackable + AllocateWeights>
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
            storagef32: vec![0.0; self.noutputs],
            storagef64: vec![0.0; self.noutputs],
        }
    }
}

impl RNNState for LSTMStateBase<f64, F64x4> {
    fn propagate<'b>(&mut self, inputs: &'b [f64]) -> &[f64] {
        self.lstm_propagate(inputs)
    }

    fn propagate32<'b>(&mut self, inputs: &'b [f32]) -> &[f32] {
        let mut inputs64: Vec<f64> = Vec::with_capacity(inputs.len());
        for i in inputs.iter() {
            inputs64.push(*i as f64);
        }
        let storagef32: *mut f32 = self.storagef32.as_mut_ptr();
        let noutputs = self.network.noutputs;
        let out = self.lstm_propagate(&inputs64);
        for idx in 0..noutputs {
            unsafe {
                *storagef32.add(idx) = *out.get_unchecked(idx) as f32;
            }
        }
        &self.storagef32[..]
    }

    fn reset(&mut self) {
        for (idx, m) in self.network.initial_memories.iter().enumerate() {
            unsafe {
                self.memories.get_unchecked_mut(idx)[..].clone_from_slice(m);
            }
        }
        for layer in self.last_activations.iter_mut() {
            for m in layer.iter_mut() {
                *m = 0.0;
            }
        }
    }
}

impl RNNState for LSTMStateBase<f32, F32x8> {
    fn propagate<'b>(&mut self, inputs: &'b [f64]) -> &[f64] {
        let mut inputs32: Vec<f32> = Vec::with_capacity(inputs.len());
        for i in inputs.iter() {
            inputs32.push(*i as f32);
        }
        let storagef64: *mut f64 = self.storagef64.as_mut_ptr();
        let noutputs = self.network.noutputs;
        let out = self.lstm_propagate(&inputs32);
        for idx in 0..noutputs {
            unsafe {
                *storagef64.add(idx) = *out.get_unchecked(idx) as f64;
            }
        }
        &self.storagef64[..]
    }

    fn propagate32<'b>(&mut self, inputs: &'b [f32]) -> &[f32] {
        self.lstm_propagate(inputs)
    }

    fn reset(&mut self) {
        for (idx, m) in self.network.initial_memories.iter().enumerate() {
            unsafe {
                self.memories.get_unchecked_mut(idx)[..].clone_from_slice(m);
            }
        }
        for layer in self.last_activations.iter_mut() {
            for m in layer.iter_mut() {
                *m = 0.0;
            }
        }
    }
}

#[inline]
fn div2_round_up(x: usize) -> usize {
    x / 2 + x % 2
}

impl<T: 'static + Clone + FromF64, Unpack: Unpackable + AllocateWeights> LSTMStateBase<T, Unpack> {
    pub fn memories(&self) -> Vec<T> {
        let mut result = vec![];
        for m in self.memories.iter() {
            for v in m.iter() {
                result.push(v.clone());
            }
        }
        result
    }
}

impl LSTMStateBase<f32, F32x8> {
    #[inline]
    pub fn lstm_propagate<'b>(&'b mut self, inputs: &[f32]) -> &'b [f32] {
        unsafe { self.lstm_propagate2(inputs) }
    }

    pub fn smallest_largest_memory(&self) -> (f32, f32) {
        let mut lowest: Option<f32> = None;
        let mut highest: Option<f32> = None;
        for vec in self.memories.iter() {
            for x in vec.iter() {
                let x: f32 = *x;
                if lowest.is_none() || lowest > Some(x) {
                    lowest = Some(x);
                }
                if highest.is_none() || highest < Some(x) {
                    highest = Some(x);
                }
            }
        }
        (lowest.unwrap_or(0.0), highest.unwrap_or(0.0))
    }

    unsafe fn lstm_propagate2<'b>(&'b mut self, inputs: &[f32]) -> &'b [f32] {
        assert_eq!(inputs.len(), self.network.ninputs);

        let zero: f32 = 0.0;

        let mut outputs1: &mut [f32] = &mut self.storage1;
        let mut outputs2: &mut [f32] = &mut self.storage2;

        outputs2[..inputs.len()].clone_from_slice(inputs);

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
        &outputs1[0..self.network.noutputs]
    }
}

impl LSTMStateBase<f64, F64x4> {
    #[inline]
    pub fn lstm_propagate<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        unsafe { self.lstm_propagate2(inputs) }
    }

    pub fn smallest_largest_memory(&self) -> (f64, f64) {
        let mut lowest: Option<f64> = None;
        let mut highest: Option<f64> = None;
        for vec in self.memories.iter() {
            for x in vec.iter() {
                let x: f64 = *x;
                if lowest.is_none() || lowest > Some(x) {
                    lowest = Some(x);
                }
                if highest.is_none() || highest < Some(x) {
                    highest = Some(x);
                }
            }
        }
        (lowest.unwrap_or(0.0), highest.unwrap_or(0.0))
    }

    unsafe fn lstm_propagate2<'b>(&'b mut self, inputs: &[f64]) -> &'b [f64] {
        if !self.network.initial_memories.is_empty() {
            assert_eq!(
                inputs.len() * self.network.initial_memories[0].len(),
                self.network.weights[0].len()
            );
        }

        let mut outputs1 = &mut self.storage1;
        let mut outputs2 = &mut self.storage2;

        outputs2[..inputs.len()].clone_from_slice(inputs);

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

impl Serialize for F32x8 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unsafe { self.vec.serialize(serializer) }
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

impl<'de> Deserialize<'de> for F32x8 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec8 = Vec8_F32::deserialize(deserializer)?;
        Ok(F32x8 { vec: vec8 })
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
                .map(|inner| vecf64_to_vecf32(inner))
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

impl LSTMNetworkF32 {
    #[inline]
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
        fn f32_network_f32_and_f64_propagate_are_same(net: LSTMNetworkF32) -> bool {
            let inputs_f32 = vec![0.0; net.num_inputs()];
            let inputs_f64 = vec![0.0; net.num_inputs()];
            let mut st1 = net.start();
            let out1 = st1.propagate(&inputs_f64);
            let mut st2 = net.start();
            let out2 = st2.propagate32(&inputs_f32);

            let out1_f32: Vec<f32> = out1.into_iter().map(|x| *x as f32).collect();
            out1_f32 == out2
        }
    }

    quickcheck! {
        fn f64_network_f32_and_f64_propagate_are_same(net: LSTMNetwork) -> bool {
            let inputs_f32 = vec![0.0; net.num_inputs()];
            let inputs_f64 = vec![0.0; net.num_inputs()];
            let mut st1 = net.start();
            let out1 = st1.propagate(&inputs_f64);
            let mut st2 = net.start();
            let out2 = st2.propagate32(&inputs_f32);

            let out1_f32: Vec<f32> = out1.into_iter().map(|x| *x as f32).collect();
            out1_f32 == out2
        }
    }

    quickcheck! {
        fn f64_lstm_state_reset_is_equivalent_to_fresh_state(net: LSTMNetwork) -> bool {
            let inputs_f64 = vec![0.0; net.num_inputs()];
            let mut st1 = net.start();
            let out1 = st1.propagate(&inputs_f64).to_vec();
            st1.reset();
            let out2 = st1.propagate(&inputs_f64).to_vec();
            out1 == out2
        }
    }

    quickcheck! {
        fn f32_lstm_state_reset_is_equivalent_to_fresh_state(net: LSTMNetworkF32) -> bool {
            let inputs_f32 = vec![0.0; net.num_inputs()];
            let mut st1 = net.start();
            let out1 = st1.propagate32(&inputs_f32).to_vec();
            st1.reset();
            let out2 = st1.propagate32(&inputs_f32).to_vec();
            out1 == out2
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
