This package contains random recurrent neural network code I'm using for random
experiments.

  * LSTM and GRU implementation
  * The LSTMv2 has an implementation of gradient descent (lstm\_v2.rs)
  * Some other experimental neural network implementations.
  * Miscellaneous utilities to detect when training is finished.

Basic example using gradient descent to train an LSTM network that outputs
01010101 etc. bitstring:

```rust
use mj_lstm::lstm_v2::*;
use rand::{thread_rng, Rng};

fn main() {
    let mut nn = LSTMv2::new(&[1, 5, 5, 1]);
    nn.randomize();
    let mut adamw = nn.adamw(AdamWConfiguration::default());
    let mut grad = nn.zero_like();

    let mut rng = thread_rng();

    loop {
        let mut st = nn.start_v2();
        let mut out: Vec<f64> = vec![0.0; 1];
        let mut score: f64 = 0.0;
        let mut show_output = rng.gen_range(1..100) == 1;
        for idx in 1..1000 {
            let t = idx as f64 / 100.0;
            st.propagate_collect_gradients_v2(&nn, &[t], &mut out);
            let wanted = if idx % 2 == 0 {
                1.0
            } else {
                0.0
            };
            let err = wanted - out[0];
            score += err * err;
            st.set_output_derivs(&nn, &[out[0] - wanted]);
            if show_output {
                println!("idx={} out={} wanted={} err={}", idx, out[0], wanted, err);
            }
        }
        score /= 1000.0;
        println!("Score: {}", score);
        grad.set_zero();
        st.backpropagate(&nn, &mut grad);
        nn.update_parameters_from_adamw_and_gradient(&grad, &mut adamw);
    }
}
```
