use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mj_lstm::lstm::*;
use mj_lstm::rnn::*;

pub fn lstm_benchmark(c: &mut Criterion) {
    let lstm = LSTMNetwork::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f64 8-100-100-8 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm).start();
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f32 8-100-100-8 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm32).start();
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm = LSTMNetwork::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f64 5-8-4-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm).start();
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f32 5-8-4-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm32).start();
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });
}

criterion_group!(benches, lstm_benchmark);
criterion_main!(benches);
