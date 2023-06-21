/*
 * Experimental implementation of "Meta Learning Backpropagation And Improving It" by
 * Louis Kirsch, Jürgen Schmidhuber
 * https://doi.org/10.48550/arXiv.2012.14905
 *
 * It's a neural network where the weights are replaced by tinier recurrent neural networks that in
 * turn implement a learning algorithm (which may or may not be backpropagation).
 *
 * As of writing of this, I've confirmed this works for simple neural networks, even through
 * several layers deep. I haven't tried massively big neural networks though.
 *
 * I haven't implemented the paper exactly as it is, but I've taken its ideas. Work in progress.
 */

use crate::simd_common::sigmoid;
use rand::{thread_rng, Rng};
use rcmaes::Vectorizable;

#[derive(Clone, Debug)]
pub struct MetaNNState {
    meta: MetaNN,
    ninputs: usize,
    noutputs: usize,
    nhiddens: Vec<usize>,
    meta_states: Vec<MetaLayerState>,
    message_matrix: MessageMatrix,
}

#[derive(Clone, Debug)]
pub struct MetaNN {
    sz: usize,
    // all the matrices are squares
    w: Vec2D<f64>,
    w_bias: Vec<f64>,
    forward_w: Vec2D<f64>,
    forward_w_bias: Vec<f64>,
    backward_w: Vec2D<f64>,
    backward_w_bias: Vec<f64>,
}

impl Vectorizable for MetaNN {
    type Context = usize;

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        let mut result: Vec<f64> = vec![];
        result.extend_from_slice(&self.w.items);
        result.extend_from_slice(&self.w_bias);
        result.extend_from_slice(&self.forward_w.items);
        result.extend_from_slice(&self.forward_w_bias);
        result.extend_from_slice(&self.backward_w.items);
        result.extend_from_slice(&self.backward_w_bias);
        (result, self.sz)
    }

    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self {
        let sz = *ctx;
        let mut result = Self::new(sz);
        let mut cursor: usize = 0;
        result.w.clone_from_slice(&vec[cursor..cursor + sz * sz]);
        cursor += sz * sz;
        result.w_bias.clone_from_slice(&vec[cursor..cursor + sz]);
        cursor += sz;
        result
            .forward_w
            .clone_from_slice(&vec[cursor..cursor + sz * sz]);
        cursor += sz * sz;
        result
            .forward_w_bias
            .clone_from_slice(&vec[cursor..cursor + sz]);
        cursor += sz;
        result
            .backward_w
            .clone_from_slice(&vec[cursor..cursor + sz * sz]);
        cursor += sz * sz;
        result
            .backward_w_bias
            .clone_from_slice(&vec[cursor..cursor + sz]);
        result
    }
}

fn randomize_vec(vec: &mut [f64]) {
    let mut rng = thread_rng();
    for v in vec.iter_mut() {
        *v = rng.gen_range(-0.01, 0.01);
    }
}

fn randomize_vec2d(vec: &mut Vec2D<f64>) {
    let mut rng = thread_rng();
    for col in 0..vec.cols() {
        for row in 0..vec.rows() {
            vec.set(row, col, rng.gen_range(-0.01, 0.01));
        }
    }
}

impl MetaNN {
    pub fn new(sz: usize) -> Self {
        let mut result = Self {
            sz,
            w: Vec2D::replicate(0.0, sz, sz),
            w_bias: vec![0.0; sz],
            forward_w: Vec2D::replicate(0.0, sz, sz),
            forward_w_bias: vec![0.0; sz],
            backward_w: Vec2D::replicate(0.0, sz, sz),
            backward_w_bias: vec![0.0; sz],
        };
        randomize_vec2d(&mut result.w);
        randomize_vec(&mut result.w_bias);
        randomize_vec2d(&mut result.forward_w);
        randomize_vec(&mut result.forward_w_bias);
        randomize_vec2d(&mut result.backward_w);
        randomize_vec(&mut result.backward_w_bias);
        result
    }

    pub fn size(&self) -> usize {
        self.sz
    }
}

#[derive(Clone, Debug)]
pub struct Vec2D<T> {
    nrows: usize,
    ncols: usize,
    items: Vec<T>,
}

impl<T: Clone> Vec2D<T> {
    pub fn replicate(item: T, rows: usize, cols: usize) -> Self {
        Self {
            nrows: rows,
            ncols: cols,
            items: vec![item; rows * cols],
        }
    }

    pub fn clone_from_slice(&mut self, slice: &[T]) {
        assert_eq!(slice.len(), self.items.len());
        self.items.clone_from_slice(slice);
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> &T {
        &self.items[x * self.ncols + y]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self.items[x * self.ncols + y] = value;
    }

    #[inline]
    pub fn modify<F>(&mut self, x: usize, y: usize, modify: F)
    where
        F: FnOnce(&mut T),
    {
        modify(&mut self.items[x * self.ncols + y]);
    }
}

#[derive(Clone, Debug)]
pub struct MetaLayerState {
    // states, or weights in a neural network. A neurons on left side and B neurons on right side.
    states: Vec2D<MetaNodeState>,     // AxB
    tmp_states: Vec2D<MetaNodeState>, // AxB
}

impl MetaLayerState {
    pub fn new(meta: &MetaNN, a: usize, b: usize) -> Self {
        Self {
            states: Vec2D::replicate(MetaNodeState::new(meta.size()), a, b),
            tmp_states: Vec2D::replicate(MetaNodeState::new(meta.size()), a, b),
        }
    }

    pub fn propagate(
        &mut self,
        meta_nn: &MetaNN,
        layer_idx: usize,
        message_matrix: &mut MessageMatrix,
    ) {
        let forward_msg: &[Message] = &message_matrix.forward_messages[layer_idx];
        let backward_msg: &[Message] = &message_matrix.backward_messages[layer_idx];
        let forward_msg_len = forward_msg.len();
        let backward_msg_len = backward_msg.len();
        assert_eq!(forward_msg_len, self.states.rows());
        assert_eq!(backward_msg_len, self.states.cols());

        let b_sz: usize = message_matrix.forward_messages[layer_idx + 1].len();

        for a_idx in 0..self.states.rows() {
            for b_idx in 0..self.states.cols() {
                for j in 0..meta_nn.size() {
                    let mut accum: f64 = meta_nn.w_bias[j];
                    // self-connections
                    for i in 0..meta_nn.size() {
                        accum += self.states.at(a_idx, b_idx).last_activations[i]
                            * (*meta_nn.w.at(i, j));
                    }
                    // (message weight matrices are computed when messages are generated, not here)
                    // forward messages
                    for i in 0..meta_nn.size() {
                        accum += forward_msg[a_idx].msg[i];
                    }
                    // backward messages
                    for i in 0..meta_nn.size() {
                        accum += backward_msg[b_idx].msg[i];
                    }
                    accum = sigmoid(accum);
                    self.tmp_states.modify(a_idx, b_idx, |last_activations| {
                        last_activations.last_activations[j] = accum;
                    });
                }
            }
        }
        std::mem::swap(&mut self.states, &mut self.tmp_states);

        // Compute new forward and backward messages.
        let new_forward_msg: &mut [Message] = &mut message_matrix.forward_messages[layer_idx + 1];
        let new_forward_msg_len: usize = new_forward_msg.len();
        assert_eq!(new_forward_msg_len, b_sz);
        assert_eq!(new_forward_msg_len, self.states.cols());

        for b_idx in 0..self.states.cols() {
            for j in 0..meta_nn.size() {
                let mut accum: f64 = meta_nn.forward_w_bias[j];
                for a_idx in 0..self.states.rows() {
                    for i in 0..meta_nn.size() {
                        accum += (*meta_nn.forward_w.at(i, j))
                            * self.states.at(a_idx, b_idx).last_activations[i];
                    }
                }
                new_forward_msg[b_idx].msg[j] = accum;
            }
        }

        if layer_idx > 0 {
            let new_backward_msg: &mut [Message] =
                &mut message_matrix.backward_messages[layer_idx - 1];
            assert_eq!(new_backward_msg.len(), forward_msg_len);

            for a_idx in 0..new_backward_msg.len() {
                for j in 0..meta_nn.size() {
                    let mut accum: f64 = meta_nn.backward_w_bias[j];
                    for b_idx in 0..self.states.cols() {
                        for i in 0..meta_nn.size() {
                            accum += (*meta_nn.backward_w.at(i, j))
                                * self.states.at(a_idx, b_idx).last_activations[i];
                        }
                    }
                    new_backward_msg[a_idx].msg[j] = accum;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MetaNodeState {
    last_activations: Vec<f64>,
}

impl MetaNodeState {
    pub fn new(sz: usize) -> Self {
        Self {
            last_activations: vec![0.0; sz],
        }
    }

    pub fn reset(&mut self) {
        for v in self.last_activations.iter_mut() {
            *v = 0.0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct MessageMatrix {
    forward_messages: Vec<Vec<Message>>,
    backward_messages: Vec<Vec<Message>>,
    nlayers: usize,
}

impl MessageMatrix {
    pub fn new(sizes: &[usize], meta_nn: &MetaNN) -> Self {
        let mut forward_messages: Vec<Vec<Message>> = Vec::with_capacity(sizes.len());
        let mut backward_messages: Vec<Vec<Message>> = Vec::with_capacity(sizes.len());
        let sz = meta_nn.size();
        for idx in 0..sizes.len() - 1 {
            let a_sz: usize = sizes[idx];
            let b_sz: usize = sizes[idx + 1];

            forward_messages.push(vec![Message::new(sz); a_sz]);
            backward_messages.push(vec![Message::new(sz); b_sz]);
        }
        forward_messages.push(vec![Message::new(sz); sizes[sizes.len() - 1]]);
        Self {
            forward_messages,
            backward_messages,
            nlayers: sizes.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Message {
    msg: Vec<f64>,
}

impl Message {
    pub fn new(sz: usize) -> Self {
        Self { msg: vec![0.0; sz] }
    }

    pub fn input_set(&mut self, inp: f64) {
        for (idx, v) in self.msg.iter_mut().enumerate() {
            if idx == 0 {
                *v = inp;
            } else {
                *v = 0.0;
            }
        }
    }
}

impl MetaNNState {
    pub fn new(ninputs: usize, nhiddens: &[usize], noutputs: usize, meta: MetaNN) -> Self {
        let mut message_matrix_sizes: Vec<usize> = Vec::with_capacity(2 + nhiddens.len());
        message_matrix_sizes.push(ninputs);
        for sz in nhiddens.iter() {
            message_matrix_sizes.push(*sz);
        }
        message_matrix_sizes.push(noutputs);
        let message_matrix = MessageMatrix::new(&message_matrix_sizes, &meta);

        let mut meta_states: Vec<MetaLayerState> = Vec::with_capacity(2 + nhiddens.len());
        for idx in 1..message_matrix_sizes.len() {
            let a = message_matrix_sizes[idx - 1];
            let b = message_matrix_sizes[idx];
            meta_states.push(MetaLayerState::new(&meta, a, b));
        }

        Self {
            ninputs,
            noutputs,
            nhiddens: nhiddens.to_vec(),
            meta,
            message_matrix,
            meta_states,
        }
    }

    pub fn propagate(&mut self, inputs: &[f64], outputs: &mut [f64]) {
        assert_eq!(inputs.len(), self.message_matrix.forward_messages[0].len());
        assert_eq!(outputs.len(), self.noutputs);

        for (idx, fm) in self.message_matrix.forward_messages[0]
            .iter_mut()
            .enumerate()
        {
            fm.input_set(inputs[idx]);
        }

        for layer_idx in 0..self.meta_states.len() {
            self.meta_states[layer_idx].propagate(&self.meta, layer_idx, &mut self.message_matrix);
        }

        for target_idx in 0..self.noutputs {
            outputs[target_idx] =
                sigmoid(self.message_matrix.forward_messages.last().unwrap()[target_idx].msg[0]);
        }
    }

    pub fn set_errors(&mut self, errors: &[f64]) {
        assert_eq!(errors.len(), self.noutputs);
        for (idx, fm) in self
            .message_matrix
            .backward_messages
            .last_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            fm.input_set(errors[idx]);
        }
    }
}
