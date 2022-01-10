use crate::rnn::{RNNState, RNN};
#[cfg(target_arch = "aarch64")]
use crate::simd_aarch64::*;
#[cfg(target_arch = "x86_64")]
use crate::simd_amd64::*;
use crate::simd_common::fast_sigmoid;
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUW {
    z: f64,
    r: f64,
    h: f64,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUWSIMD {
    zr: F64x2,
    h: f64,
}

pub trait GRUWLike {
    fn z(&self) -> f64;
    fn r(&self) -> f64;
    fn h(&self) -> f64;
    fn set_z(&mut self, z: f64);
    fn set_r(&mut self, r: f64);
    fn set_h(&mut self, h: f64);
    fn zero() -> Self;
}

impl GRUWLike for GRUWSIMD {
    #[inline]
    fn z(&self) -> f64 {
        self.zr.v1()
    }

    #[inline]
    fn r(&self) -> f64 {
        self.zr.v2()
    }

    #[inline]
    fn h(&self) -> f64 {
        self.h
    }

    #[inline]
    fn set_z(&mut self, z: f64) {
        *self.zr.v1_mut() = z;
    }

    #[inline]
    fn set_r(&mut self, r: f64) {
        *self.zr.v2_mut() = r;
    }

    #[inline]
    fn set_h(&mut self, h: f64) {
        self.h = h;
    }

    #[inline]
    fn zero() -> Self {
        GRUWSIMD {
            zr: unsafe { F64x2::new(0.0, 0.0) },
            h: 0.0,
        }
    }
}

impl GRUWLike for GRUW {
    #[inline]
    fn z(&self) -> f64 {
        self.z
    }

    #[inline]
    fn r(&self) -> f64 {
        self.r
    }

    #[inline]
    fn h(&self) -> f64 {
        self.h
    }

    #[inline]
    fn set_z(&mut self, z: f64) {
        self.z = z;
    }

    #[inline]
    fn set_r(&mut self, r: f64) {
        self.r = r;
    }

    #[inline]
    fn set_h(&mut self, h: f64) {
        self.h = h;
    }

    #[inline]
    fn zero() -> Self {
        GRUW {
            z: 0.0,
            r: 0.0,
            h: 0.0,
        }
    }
}

pub type GRUNetwork = GRUNetworkBase<GRUWSIMD>;
pub type GRUNetworkSIMD = GRUNetworkBase<GRUWSIMD>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUNetworkBase<G> {
    pub(crate) i_to_h_weights: Vec<Vec<G>>,
    pub(crate) h_to_h_weights: Vec<Vec<G>>,
    pub(crate) biases: Vec<Vec<G>>,
    pub(crate) output_weights: Vec<f64>,
    pub(crate) output_biases: Vec<f64>,
    pub(crate) layer_sizes: Vec<usize>,
    pub(crate) widest_layer: usize,
}

pub type GRUState = GRUStateBase<GRUWSIMD>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUStateBase<G> {
    gru: GRUNetworkBase<G>,
    memories: Vec<Vec<f64>>,
    memories2: Vec<Vec<f64>>,
    outputs: Vec<f64>,
    storage1: Vec<f64>,
    storage2: Vec<f64>,
}

#[cfg(test)]
impl quickcheck::Arbitrary for GRUNetwork {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        use rand::{thread_rng, Rng};

        let nlayers: usize = std::cmp::max(3, usize::arbitrary(g) % 3);
        let mut layer_sizes: Vec<usize> = vec![];
        for _ in 0..nlayers {
            layer_sizes.push(5);
            //layer_sizes.push(std::cmp::max(1, usize::arbitrary(g) % 3));
        }

        // TODO: use quickcheck::Gen not thread_rng
        let mut rng = thread_rng();

        let mut network = GRUNetwork::new(&layer_sizes);
        let (mut vec, ctx) = network.to_vec();
        for v in vec.iter_mut() {
            *v += rng.gen_range(-1.0, 1.0);
        }
        GRUNetwork::from_vec(&vec, &ctx)
    }
}

impl<G: GRUWLike + Clone> Vectorizable for GRUNetworkBase<G> {
    type Context = Vec<usize>;

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut out = vec![];
        for w in self.i_to_h_weights.iter() {
            for gruw in w.iter() {
                out.push(gruw.z());
                out.push(gruw.r());
                out.push(gruw.h());
            }
        }
        for w in self.h_to_h_weights.iter() {
            for gruw in w.iter() {
                out.push(gruw.z());
                out.push(gruw.r());
                out.push(gruw.h());
            }
        }
        for b in self.biases.iter() {
            for gruw in b.iter() {
                out.push(gruw.z());
                out.push(gruw.r());
                out.push(gruw.h());
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
        let mut network = GRUNetworkBase::<G>::new(ctx);
        for w in network.i_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.set_z(vec[cursor]);
                gruw.set_r(vec[cursor + 1]);
                gruw.set_h(vec[cursor + 2]);
                cursor += 3;
            }
        }
        for w in network.h_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.set_z(vec[cursor]);
                gruw.set_r(vec[cursor + 1]);
                gruw.set_h(vec[cursor + 2]);
                cursor += 3;
            }
        }
        for b in network.biases.iter_mut() {
            for gruw in b.iter_mut() {
                gruw.set_z(vec[cursor]);
                gruw.set_r(vec[cursor + 1]);
                gruw.set_h(vec[cursor + 2]);
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

impl<G: GRUWLike + Clone> GRUNetworkBase<G> {
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
            i_to_h_weights.push(vec![G::zero(); prev_layer * layer]);
            h_to_h_weights.push(vec![G::zero(); layer * layer]);
            biases.push(vec![G::zero(); layer]);
            widest_layer = std::cmp::max(widest_layer, layer);
        }

        let output_weights =
            vec![0.0; layer_sizes[layer_sizes.len() - 1] * layer_sizes[layer_sizes.len() - 2]];
        let output_biases = vec![0.0; *layer_sizes.last().unwrap()];

        GRUNetworkBase {
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

impl<G: GRUWLike + Clone> RNN for GRUNetworkBase<G> {
    type RNNState = GRUStateBase<G>;

    fn start(&self) -> Self::RNNState {
        let mut memories: Vec<Vec<f64>> = Vec::with_capacity(self.layer_sizes.len() - 2);
        for b in self.biases.iter() {
            memories.push(vec![0.0; b.len()]);
        }

        let memories2 = memories.clone();

        GRUStateBase {
            gru: self.clone(),
            memories,
            memories2,
            outputs: vec![],
            storage1: vec![],
            storage2: vec![],
        }
    }
}

trait Prop {
    fn prop64<'a>(s: &mut GRUStateBase<Self>, inputs: &[f64])
    where
        Self: Sized;
}

impl<G: GRUWLike + Clone + Prop> RNNState for GRUStateBase<G> {
    fn reset(&mut self) {
        for v in self.memories.iter_mut() {
            for v2 in v.iter_mut() {
                *v2 = 0.0;
            }
        }
    }

    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [f64]) -> &'a [f64] {
        G::prop64(self, inputs);
        &self.outputs
    }

    fn propagate32<'a, 'b>(&'a mut self, _inputs: &'b [f32]) -> &'a [f32] {
        unimplemented!();
    }
}

impl Prop for GRUWSIMD {
    fn prop64<'a>(s: &mut GRUStateBase<Self>, inputs: &[f64]) {
        unsafe { prop64_unsafe(s, inputs) }
    }
}

#[inline]
unsafe fn prop64_unsafe(s: &mut GRUStateBase<GRUWSIMD>, inputs: &[f64]) {
    assert!(inputs.len() == s.gru.num_inputs());

    unsafe {
        s.outputs.resize(s.gru.num_outputs(), 0.0);
        s.storage1.resize(s.gru.widest_layer, 0.0);
        s.storage2.resize(s.gru.widest_layer, 0.0);

        for idx in 0..inputs.len() {
            *s.storage1.get_unchecked_mut(idx) = *inputs.get_unchecked(idx);
        }

        for layer_idx in 0..s.gru.biases.len() {
            let previous_layer_size = *s.gru.layer_sizes.get_unchecked(layer_idx);
            let layer_size = *s.gru.layer_sizes.get_unchecked(layer_idx + 1);

            for target_offset in 0..layer_size / 2 {
                let mut zr: F64x4 = F64x4::from_F64x2(
                    s.gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_offset * 2)
                        .zr,
                    s.gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_offset * 2 + 1)
                        .zr,
                );
                for source_idx in 0..previous_layer_size {
                    zr.mul_add_scalar(
                        *s.storage1.get_unchecked(source_idx),
                        F64x4::from_F64x2(
                            s.gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(
                                    source_idx + (target_offset * 2) * previous_layer_size,
                                )
                                .zr,
                            s.gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(
                                    source_idx + (target_offset * 2 + 1) * previous_layer_size,
                                )
                                .zr,
                        ),
                    );
                    zr.mul_add_scalar(
                        *s.memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx),
                        F64x4::from_F64x2(
                            s.gru
                                .h_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + (target_offset * 2) * layer_size)
                                .zr,
                            s.gru
                                .h_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + (target_offset * 2 + 1) * layer_size)
                                .zr,
                        ),
                    );
                }
                zr.fast_sigmoid();

                let mut ht = F64x2::new(
                    s.gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_offset * 2)
                        .h(),
                    s.gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_offset * 2 + 1)
                        .h(),
                );

                for source_idx in 0..previous_layer_size {
                    ht.mul_add_scalar(
                        *s.storage1.get_unchecked(source_idx),
                        F64x2::new(
                            s.gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(
                                    source_idx + (target_offset * 2) * previous_layer_size,
                                )
                                .h(),
                            s.gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(
                                    source_idx + (target_offset * 2 + 1) * previous_layer_size,
                                )
                                .h(),
                        ),
                    );
                }
                for source_idx in 0..layer_size {
                    let mut tmp = F64x2::new(
                        s.gru
                            .h_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + (target_offset * 2) * layer_size)
                            .h(),
                        s.gru
                            .h_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + (target_offset * 2 + 1) * layer_size)
                            .h(),
                    );
                    *tmp.v1_mut() *= zr.v2();
                    *tmp.v2_mut() *= zr.v4();
                    ht.mul_add_scalar(
                        *s.memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx),
                        tmp,
                    );
                }
                ht.fast_sigmoid();
                ht.mul_add_scalar_scalar(2.0, -1.0);

                let ht_final_1 = (1.0 - zr.v1())
                    * s.memories
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_offset * 2)
                    + zr.v1() * ht.v1();
                *s.memories2
                    .get_unchecked_mut(layer_idx)
                    .get_unchecked_mut(target_offset * 2) = ht_final_1;
                *s.storage2.get_unchecked_mut(target_offset * 2) = ht_final_1;

                let ht_final_2 = (1.0 - zr.v3())
                    * s.memories
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_offset * 2 + 1)
                    + zr.v3() * ht.v2();
                *s.memories2
                    .get_unchecked_mut(layer_idx)
                    .get_unchecked_mut(target_offset * 2 + 1) = ht_final_2;
                *s.storage2.get_unchecked_mut(target_offset * 2 + 1) = ht_final_2;
            }
            if layer_size % 2 == 1 {
                let target_idx = layer_size - 1;
                // compute z[t]
                let mut zt = s
                    .gru
                    .biases
                    .get_unchecked(layer_idx)
                    .get_unchecked(target_idx)
                    .z();
                for source_idx in 0..previous_layer_size {
                    zt += s
                        .gru
                        .i_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * previous_layer_size)
                        .z()
                        * s.storage1.get_unchecked(source_idx);
                }
                for source_idx in 0..layer_size {
                    zt += s
                        .gru
                        .h_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * layer_size)
                        .z()
                        * (*s
                            .memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx));
                }
                zt = fast_sigmoid(zt);

                // compute r[t]
                let mut rt = s
                    .gru
                    .biases
                    .get_unchecked(layer_idx)
                    .get_unchecked(target_idx)
                    .r();
                for source_idx in 0..previous_layer_size {
                    rt += s
                        .gru
                        .i_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * previous_layer_size)
                        .r()
                        * (*s.storage1.get_unchecked(source_idx));
                }
                for source_idx in 0..layer_size {
                    rt += s
                        .gru
                        .h_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * layer_size)
                        .r()
                        * s.memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx);
                }
                rt = fast_sigmoid(rt);

                // compute h^[t]
                let mut ht = s
                    .gru
                    .biases
                    .get_unchecked(layer_idx)
                    .get_unchecked(target_idx)
                    .h();
                for source_idx in 0..previous_layer_size {
                    ht += s
                        .gru
                        .i_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * previous_layer_size)
                        .h()
                        * (*s.storage1.get_unchecked(source_idx));
                }
                for source_idx in 0..layer_size {
                    ht += s
                        .gru
                        .h_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * layer_size)
                        .h()
                        * (*s
                            .memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx))
                        * rt;
                }
                ht = fast_sigmoid(ht) * 2.0 - 1.0;

                // compute h[t] (next hidden state, also output)
                let ht_final = (1.0 - zt)
                    * (*s
                        .memories
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx))
                    + zt * ht;
                *s.memories2
                    .get_unchecked_mut(layer_idx)
                    .get_unchecked_mut(target_idx) = ht_final;
                *s.storage2.get_unchecked_mut(target_idx) = ht_final;
            }
            // SIMD 1 loop
            /*
            for target_idx in 0..layer_size {
                let mut zr: F64x2 = s
                    .gru
                    .biases
                    .get_unchecked(layer_idx)
                    .get_unchecked(target_idx)
                    .zr;
                for source_idx in 0..previous_layer_size {
                    zr.mul_add_scalar(
                        *s.storage1.get_unchecked(source_idx),
                        s.gru
                            .i_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * previous_layer_size)
                            .zr,
                    );
                    zr.mul_add_scalar(
                        *s.memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx),
                        s.gru
                            .h_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * layer_size)
                            .zr,
                    );
                }
                zr.fast_sigmoid();

                let mut ht = s
                    .gru
                    .biases
                    .get_unchecked(layer_idx)
                    .get_unchecked(target_idx)
                    .h();
                for source_idx in 0..previous_layer_size {
                    ht += s
                        .gru
                        .i_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * previous_layer_size)
                        .h()
                        * (*s.storage1.get_unchecked(source_idx));
                }
                for source_idx in 0..layer_size {
                    ht += s
                        .gru
                        .h_to_h_weights
                        .get_unchecked(layer_idx)
                        .get_unchecked(source_idx + target_idx * layer_size)
                        .h()
                        * (*s
                            .memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx))
                        * zr.v2();
                }
                ht = fast_sigmoid(ht) * 2.0 - 1.0;

                let ht_final = (1.0 - zr.v1())
                    * s.memories
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                    + zr.v1() * ht;
                *s.memories2
                    .get_unchecked_mut(layer_idx)
                    .get_unchecked_mut(target_idx) = ht_final;
                *s.storage2.get_unchecked_mut(target_idx) = ht_final;
            }
            */
            std::mem::swap(&mut s.storage1, &mut s.storage2);
        }
        for target_idx in 0..s.gru.num_outputs() {
            let mut sum = *s.gru.output_biases.get_unchecked(target_idx);
            let sz = *s.gru.layer_sizes.get_unchecked(s.gru.layer_sizes.len() - 2);
            for source_idx in 0..sz {
                sum += *s
                    .gru
                    .output_weights
                    .get_unchecked(source_idx + target_idx * sz)
                    * (*s.storage1.get_unchecked(source_idx));
            }
            sum = fast_sigmoid(sum);
            *s.outputs.get_unchecked_mut(target_idx) = sum;
        }
        std::mem::swap(&mut s.memories, &mut s.memories2);
    }
}

impl Prop for GRUW {
    fn prop64<'a>(s: &mut GRUStateBase<Self>, inputs: &[f64]) {
        assert!(inputs.len() == s.gru.num_inputs());

        unsafe {
            s.outputs.resize(s.gru.num_outputs(), 0.0);
            s.storage1.resize(s.gru.widest_layer, 0.0);
            s.storage2.resize(s.gru.widest_layer, 0.0);

            for idx in 0..inputs.len() {
                s.storage1[idx] = inputs[idx];
            }

            for layer_idx in 0..s.gru.biases.len() {
                let previous_layer_size = *s.gru.layer_sizes.get_unchecked(layer_idx);
                let layer_size = *s.gru.layer_sizes.get_unchecked(layer_idx + 1);
                for target_idx in 0..layer_size {
                    // compute z[t]
                    let mut zt = s
                        .gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                        .z();
                    for source_idx in 0..previous_layer_size {
                        zt += s
                            .gru
                            .i_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * previous_layer_size)
                            .z()
                            * s.storage1.get_unchecked(source_idx);
                    }
                    for source_idx in 0..layer_size {
                        zt += s
                            .gru
                            .h_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * layer_size)
                            .z()
                            * (*s
                                .memories
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx));
                    }
                    zt = fast_sigmoid(zt);

                    // compute r[t]
                    let mut rt = s
                        .gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                        .r();
                    for source_idx in 0..previous_layer_size {
                        rt += s
                            .gru
                            .i_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * previous_layer_size)
                            .r()
                            * (*s.storage1.get_unchecked(source_idx));
                    }
                    for source_idx in 0..layer_size {
                        rt += s
                            .gru
                            .h_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * layer_size)
                            .r()
                            * s.memories
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx);
                    }
                    rt = fast_sigmoid(rt);

                    // compute h^[t]
                    let mut ht = s
                        .gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                        .h();
                    for source_idx in 0..previous_layer_size {
                        ht += s
                            .gru
                            .i_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * previous_layer_size)
                            .h()
                            * (*s.storage1.get_unchecked(source_idx));
                    }
                    for source_idx in 0..layer_size {
                        ht += s
                            .gru
                            .h_to_h_weights
                            .get_unchecked(layer_idx)
                            .get_unchecked(source_idx + target_idx * layer_size)
                            .h()
                            * (*s
                                .memories
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx))
                            * rt;
                    }
                    ht = fast_sigmoid(ht) * 2.0 - 1.0;

                    // compute h[t] (next hidden state, also output)
                    let ht_final = (1.0 - zt)
                        * (*s
                            .memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(target_idx))
                        + zt * ht;
                    *s.memories2
                        .get_unchecked_mut(layer_idx)
                        .get_unchecked_mut(target_idx) = ht_final;
                    *s.storage2.get_unchecked_mut(target_idx) = ht_final;
                }
                std::mem::swap(&mut s.storage1, &mut s.storage2);
            }
            for target_idx in 0..s.gru.num_outputs() {
                let mut sum = *s.gru.output_biases.get_unchecked(target_idx);
                let sz = *s.gru.layer_sizes.get_unchecked(s.gru.layer_sizes.len() - 2);
                for source_idx in 0..sz {
                    sum += *s
                        .gru
                        .output_weights
                        .get_unchecked(source_idx + target_idx * sz)
                        * (*s.storage1.get_unchecked(source_idx));
                }
                sum = fast_sigmoid(sum);
                *s.outputs.get_unchecked_mut(target_idx) = sum;
            }
            std::mem::swap(&mut s.memories, &mut s.memories2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck! {
        fn to_vec_from_vec_id(network: GRUNetwork) -> bool {
            let (vec, ctx) = network.to_vec();
            let new_network = GRUNetwork::from_vec(&vec, &ctx);
            new_network == network
        }
    }

    quickcheck! {
        fn simd_and_nonsimd_give_same_results(network: GRUNetwork) -> bool {
            let (vec, ctx) = network.to_vec();
            let network2: GRUNetworkSIMD = GRUNetworkSIMD::from_vec(&vec, &ctx);

            let mut inputs: Vec<f64> = Vec::with_capacity(network.num_inputs());
            for i in 0..network.num_inputs() {
                inputs.push(i as f64 * 0.1 - 0.2);
            }
            let mut st1 = network.start();
            let mut st2 = network2.start();

            st1.propagate(&inputs);
            st1.propagate(&inputs);
            let output1 = st1.propagate(&inputs);

            st2.propagate(&inputs);
            st2.propagate(&inputs);
            let output2 = st2.propagate(&inputs);

            assert_eq!(output1.len(), output2.len());

            for idx in 0..output1.len() {
                assert!((output1[idx] - output2[idx]).abs() < 0.1);
            }
            true
        }
    }
}
