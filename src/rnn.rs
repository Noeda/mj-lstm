pub trait RNNState {
    fn propagate<'a, 'b>(&'a mut self, inputs: &'b [f64]) -> &'a [f64];
}

pub trait RNN {
    type RNNState;

    fn start(&self) -> Self::RNNState;
}
