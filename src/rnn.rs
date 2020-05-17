pub trait RNNState {
    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [f64]) -> &'a [f64];
    fn propagate32<'a, 'b>(&'a mut self, inputs: &'b [f32]) -> &'a [f32];
    fn reset(&mut self);
}

pub trait RNN {
    type RNNState;

    fn start(&self) -> Self::RNNState;
}
