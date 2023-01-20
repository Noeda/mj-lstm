use crate::lstm::FromF64;
use crate::rnn::{RNNState, RNN};
use mj_autograd::*;
use num::traits::{One, Zero};
use rcmaes::Vectorizable;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub(crate) struct GRUW<T> {
    z_raw: T,
    r_raw: T,
    h_raw: T,
}

impl<T: FromF64 + Clone + One + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>
    GRUW<T>
{
    fn zero() -> Self {
        GRUW {
            z_raw: T::from_f64(0.0),
            r_raw: T::from_f64(0.0),
            h_raw: T::from_f64(0.0),
        }
    }

    fn walk<F>(&mut self, mut callback: F)
    where
        F: FnMut(&mut T),
    {
        callback(&mut self.z_raw);
        callback(&mut self.r_raw);
        callback(&mut self.h_raw);
    }

    fn to_f64(&self) -> GRUW<f64> {
        GRUW {
            z_raw: self.z_raw.to_f64(),
            r_raw: self.r_raw.to_f64(),
            h_raw: self.h_raw.to_f64(),
        }
    }

    #[inline]
    fn z(&self) -> T {
        self.z_raw.clone()
    }

    #[inline]
    fn r(&self) -> T {
        self.r_raw.clone()
    }

    #[inline]
    fn h(&self) -> T {
        self.h_raw.clone()
    }

    #[inline]
    fn set_z(&mut self, z: T) {
        self.z_raw = z;
    }

    #[inline]
    fn set_r(&mut self, r: T) {
        self.r_raw = r;
    }

    #[inline]
    fn set_h(&mut self, h: T) {
        self.h_raw = h;
    }
}

impl<T: FromF64 + Clone + Zero + One> GRUW<Reverse<T>> {
    fn from_f64(f: GRUW<f64>, tape: Tape<T>) -> Self {
        GRUW {
            z_raw: Reverse::reversible(T::from_f64(f.z_raw), tape.clone()),
            r_raw: Reverse::reversible(T::from_f64(f.r_raw), tape.clone()),
            h_raw: Reverse::reversible(T::from_f64(f.h_raw), tape),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUNetworkBase<T> {
    pub(crate) i_to_h_weights: Vec<Vec<GRUW<T>>>,
    pub(crate) h_to_h_weights: Vec<Vec<GRUW<T>>>,
    pub(crate) biases: Vec<Vec<GRUW<T>>>,
    pub(crate) output_weights: Vec<T>,
    pub(crate) output_biases: Vec<T>,
    pub(crate) layer_sizes: Vec<usize>,
    pub(crate) widest_layer: usize,
}

pub type GRUNetwork = GRUNetworkBase<f64>;
pub type GRUNetworkF32 = GRUNetworkBase<f32>;
pub type GRUNetworkGradient = GRUNetworkBase<Reverse<f64>>;
pub type GRUNetworkGradientF32 = GRUNetworkBase<Reverse<f32>>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GRUStateBase<T> {
    gru: GRUNetworkBase<T>,
    memories: Vec<Vec<T>>,
    memories2: Vec<Vec<T>>,
    outputs: Vec<T>,
    storage1: Vec<T>,
    storage2: Vec<T>,
}

impl<
        T: FromF64
            + Clone
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + One
            + Zero,
    > Vectorizable for GRUNetworkBase<T>
{
    type Context = Vec<usize>;

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut out = vec![];
        for w in self.i_to_h_weights.iter() {
            for gruw in w.iter() {
                out.push(gruw.z().to_f64());
                out.push(gruw.r().to_f64());
                out.push(gruw.h().to_f64());
            }
        }
        for w in self.h_to_h_weights.iter() {
            for gruw in w.iter() {
                out.push(gruw.z().to_f64());
                out.push(gruw.r().to_f64());
                out.push(gruw.h().to_f64());
            }
        }
        for b in self.biases.iter() {
            for gruw in b.iter() {
                out.push(gruw.z().to_f64());
                out.push(gruw.r().to_f64());
                out.push(gruw.h().to_f64());
            }
        }
        for w in self.output_weights.iter() {
            out.push(w.to_f64());
        }
        for b in self.output_biases.iter() {
            out.push(b.to_f64());
        }
        (out, self.layer_sizes.clone())
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let mut cursor = 0;
        let mut network = GRUNetworkBase::<T>::new(ctx);
        for w in network.i_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.set_z(T::from_f64(vec[cursor]));
                gruw.set_r(T::from_f64(vec[cursor + 1]));
                gruw.set_h(T::from_f64(vec[cursor + 2]));
                cursor += 3;
            }
        }
        for w in network.h_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.set_z(T::from_f64(vec[cursor]));
                gruw.set_r(T::from_f64(vec[cursor + 1]));
                gruw.set_h(T::from_f64(vec[cursor + 2]));
                cursor += 3;
            }
        }
        for b in network.biases.iter_mut() {
            for gruw in b.iter_mut() {
                gruw.set_z(T::from_f64(vec[cursor]));
                gruw.set_r(T::from_f64(vec[cursor + 1]));
                gruw.set_h(T::from_f64(vec[cursor + 2]));
                cursor += 3;
            }
        }
        for w in network.output_weights.iter_mut() {
            *w = T::from_f64(vec[cursor]);
            cursor += 1;
        }
        for b in network.output_biases.iter_mut() {
            *b = T::from_f64(vec[cursor]);
            cursor += 1;
        }
        assert!(cursor == vec.len());
        network
    }
}

impl<
        T: FromF64
            + Clone
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + One
            + Zero,
    > GRUNetworkBase<Reverse<T>>
{
    pub fn from_f64(net: &GRUNetwork, tape: Tape<T>) -> Self {
        let i_to_h_weights = net
            .i_to_h_weights
            .iter()
            .map(|v| {
                v.iter()
                    .map(|w| GRUW::<Reverse<T>>::from_f64(*w, tape.clone()))
                    .collect()
            })
            .collect();
        let h_to_h_weights = net
            .h_to_h_weights
            .iter()
            .map(|v| {
                v.iter()
                    .map(|w| GRUW::<Reverse<T>>::from_f64(*w, tape.clone()))
                    .collect()
            })
            .collect();
        let biases = net
            .biases
            .iter()
            .map(|v| {
                v.iter()
                    .map(|w| GRUW::<Reverse<T>>::from_f64(*w, tape.clone()))
                    .collect()
            })
            .collect();
        let output_weights = net
            .output_weights
            .iter()
            .map(|w| Reverse::<T>::reversible(T::from_f64(*w), tape.clone()))
            .collect();
        let output_biases = net
            .output_biases
            .iter()
            .map(|w| Reverse::<T>::reversible(T::from_f64(*w), tape.clone()))
            .collect();

        GRUNetworkBase {
            i_to_h_weights,
            h_to_h_weights,
            biases,
            output_weights,
            output_biases,
            layer_sizes: net.layer_sizes.clone(),
            widest_layer: net.widest_layer,
        }
    }
}

impl<
        T: FromF64
            + Clone
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + One
            + Zero,
    > GRUNetworkBase<T>
{
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

        let output_weights = vec![
            T::from_f64(0.0);
            layer_sizes[layer_sizes.len() - 1]
                * layer_sizes[layer_sizes.len() - 2]
        ];
        let output_biases = vec![T::from_f64(0.0); *layer_sizes.last().unwrap()];

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

    pub fn to_f64(&self) -> GRUNetwork {
        let i_to_h_weights = self
            .i_to_h_weights
            .iter()
            .map(|v| v.iter().map(|w| w.to_f64()).collect())
            .collect();
        let h_to_h_weights = self
            .h_to_h_weights
            .iter()
            .map(|v| v.iter().map(|w| w.to_f64()).collect())
            .collect();
        let biases = self
            .biases
            .iter()
            .map(|v| v.iter().map(|w| w.to_f64()).collect())
            .collect();
        let output_weights = self.output_weights.iter().map(|w| w.to_f64()).collect();
        let output_biases = self.output_biases.iter().map(|w| w.to_f64()).collect();

        GRUNetworkBase {
            i_to_h_weights,
            h_to_h_weights,
            biases,
            output_weights,
            output_biases,
            layer_sizes: self.layer_sizes.clone(),
            widest_layer: self.widest_layer,
        }
    }

    pub fn walk<F>(&mut self, mut callback: F)
    where
        F: FnMut(&mut T),
    {
        for w in self.i_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.walk(&mut callback);
            }
        }
        for w in self.h_to_h_weights.iter_mut() {
            for gruw in w.iter_mut() {
                gruw.walk(&mut callback);
            }
        }
        for b in self.biases.iter_mut() {
            for gruw in b.iter_mut() {
                gruw.walk(&mut callback);
            }
        }
        for w in self.output_weights.iter_mut() {
            callback(w);
        }
        for b in self.output_biases.iter_mut() {
            callback(b);
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

impl<T: FromF64 + Clone> RNN for GRUNetworkBase<T> {
    type RNNState = GRUStateBase<T>;

    fn start(&self) -> Self::RNNState {
        let mut memories: Vec<Vec<T>> = Vec::with_capacity(self.layer_sizes.len() - 2);
        for b in self.biases.iter() {
            memories.push(vec![T::from_f64(0.0); b.len()]);
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

impl<
        T: FromF64
            + Clone
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + One
            + Zero,
    > RNNState for GRUStateBase<T>
{
    type InputType = T;
    type OutputType = T;

    fn reset(&mut self) {
        for v in self.memories.iter_mut() {
            for v2 in v.iter_mut() {
                *v2 = T::from_f64(0.0);
            }
        }
    }

    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [T]) -> &'a [T] {
        self.prop64(inputs);
        &self.outputs
    }

    fn propagate32<'a, 'b>(&'a mut self, _inputs: &'b [f32]) -> &'a [f32] {
        unimplemented!();
    }
}

impl<
        T: FromF64
            + Clone
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + One
            + Zero,
    > GRUStateBase<T>
{
    pub fn memories(&self) -> Vec<T> {
        let mut result = vec![];
        for m in self.memories.iter() {
            result.extend_from_slice(m);
        }
        result
    }

    pub fn set_memories(&mut self, memories: &[T]) {
        let mut cursor = 0;
        for m in self.memories.iter_mut() {
            for m2 in m.iter_mut() {
                *m2 = memories[cursor].clone();
                cursor += 1;
            }
        }
    }

    fn prop64<'a>(&mut self, inputs: &[T]) {
        assert!(inputs.len() == self.gru.layer_sizes[0]);
        unsafe {
            self.outputs
                .resize(self.gru.num_outputs(), T::from_f64(0.0));
            self.storage1
                .resize(self.gru.widest_layer, T::from_f64(0.0));
            self.storage2
                .resize(self.gru.widest_layer, T::from_f64(0.0));

            for idx in 0..inputs.len() {
                *self.storage1.get_unchecked_mut(idx) = inputs.get_unchecked(idx).clone();
            }

            for layer_idx in 0..self.gru.biases.len() {
                let previous_layer_size = *self.gru.layer_sizes.get_unchecked(layer_idx);
                let layer_size = *self.gru.layer_sizes.get_unchecked(layer_idx + 1);
                for target_idx in 0..layer_size {
                    // compute z[t]
                    let mut zt = self
                        .gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                        .z();
                    for source_idx in 0..previous_layer_size {
                        zt = zt
                            + self
                                .gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + target_idx * previous_layer_size)
                                .z()
                                .clone()
                                * self.storage1.get_unchecked(source_idx).clone();
                    }
                    for source_idx in 0..layer_size {
                        zt = zt
                            + self
                                .gru
                                .h_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + target_idx * layer_size)
                                .z()
                                * self
                                    .memories
                                    .get_unchecked(layer_idx)
                                    .get_unchecked(source_idx)
                                    .clone();
                    }
                    zt = zt.fast_sigmoid();

                    // compute r[t]
                    let mut rt = self
                        .gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                        .r()
                        .clone();
                    for source_idx in 0..previous_layer_size {
                        rt = rt
                            + self
                                .gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + target_idx * previous_layer_size)
                                .r()
                                .clone()
                                * self.storage1.get_unchecked(source_idx).clone();
                    }
                    for source_idx in 0..layer_size {
                        rt = rt
                            + self
                                .gru
                                .h_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + target_idx * layer_size)
                                .r()
                                .clone()
                                * self
                                    .memories
                                    .get_unchecked(layer_idx)
                                    .get_unchecked(source_idx)
                                    .clone();
                    }
                    rt = rt.fast_sigmoid();

                    // compute h^[t]
                    let mut ht = self
                        .gru
                        .biases
                        .get_unchecked(layer_idx)
                        .get_unchecked(target_idx)
                        .h();
                    for source_idx in 0..previous_layer_size {
                        ht = ht
                            + self
                                .gru
                                .i_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + target_idx * previous_layer_size)
                                .h()
                                .clone()
                                * self.storage1.get_unchecked(source_idx).clone();
                    }
                    for source_idx in 0..layer_size {
                        ht = ht
                            + self
                                .gru
                                .h_to_h_weights
                                .get_unchecked(layer_idx)
                                .get_unchecked(source_idx + target_idx * layer_size)
                                .h()
                                .clone()
                                * self
                                    .memories
                                    .get_unchecked(layer_idx)
                                    .get_unchecked(source_idx)
                                    .clone()
                                * rt.clone();
                    }
                    ht = ht.fast_sigmoid() * T::from_f64(2.0) - T::from_f64(1.0);

                    // compute h[t] (next hidden state, also output)
                    let ht_final = (T::from_f64(1.0) - zt.clone())
                        * self
                            .memories
                            .get_unchecked(layer_idx)
                            .get_unchecked(target_idx)
                            .clone()
                        + zt.clone() * ht;
                    *self
                        .memories2
                        .get_unchecked_mut(layer_idx)
                        .get_unchecked_mut(target_idx) = ht_final.clone();
                    *self.storage2.get_unchecked_mut(target_idx) = ht_final;
                }
                std::mem::swap(&mut self.storage1, &mut self.storage2);
            }
            for target_idx in 0..self.gru.num_outputs() {
                let mut sum = self.gru.output_biases.get_unchecked(target_idx).clone();
                let sz = *self
                    .gru
                    .layer_sizes
                    .get_unchecked(self.gru.layer_sizes.len() - 2);
                for source_idx in 0..sz {
                    sum = sum
                        + self
                            .gru
                            .output_weights
                            .get_unchecked(source_idx + target_idx * sz)
                            .clone()
                            * self.storage1.get_unchecked(source_idx).clone();
                }
                sum = sum.fast_sigmoid();
                *self.outputs.get_unchecked_mut(target_idx) = sum;
            }
            std::mem::swap(&mut self.memories, &mut self.memories2);
        }
    }
}

#[cfg(test)]
impl quickcheck::Arbitrary for GRUNetwork {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let nlayers: usize = std::cmp::max(2, usize::arbitrary(g) % 20);
        let mut layer_sizes: Vec<usize> = vec![];
        for _ in 0..nlayers {
            layer_sizes.push(std::cmp::max(1, usize::arbitrary(g) % 30));
        }

        GRUNetwork::new(&layer_sizes)
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

    // These tests are just to make sure GRU can be run with all the four types:
    // f64
    // f32
    // Reverse<f64>
    // Reverse<f32>
    fn f64_smoke_test() {
        let network = GRUNetwork::new(&[2, 3, 4, 5]);
        let mut st = network.start();
        let _out = st.propagate(&[0.1, 0.2]);
    }

    fn f32_smoke_test() {
        let network = GRUNetworkF32::new(&[2, 3, 4, 5]);
        let mut st = network.start();
        let _out = st.propagate(&[0.1, 0.2]);
    }

    fn reverse_f64_smoke_test() {
        let network = GRUNetworkGradient::new(&[2, 3, 4, 5]);
        let mut st = network.start();
        let _out = st.propagate(&[Reverse::<f64>::from_f64(0.1), Reverse::<f64>::from_f64(0.2)]);
    }

    fn reverse_f32_smoke_test() {
        let network = GRUNetworkGradientF32::new(&[2, 3, 4, 5]);
        let mut st = network.start();
        let _out = st.propagate(&[Reverse::<f32>::from_f64(0.1), Reverse::<f32>::from_f64(0.2)]);
    }
}
