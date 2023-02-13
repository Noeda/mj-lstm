use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mj_lstm::gru::*;
use mj_lstm::lstm::*;
use mj_lstm::rnn::*;
use mj_lstm::simple::*;

pub fn feedforward_nn_benchmark(c: &mut Criterion) {
    let simple = SimpleNN::new(&[8, 100, 100, 8]);
    c.bench_function("feedforward 8-100-100-8 propagate", |b| {
        let mut storage1: Vec<f64> = vec![0.0; 100];
        let mut storage2: Vec<f64> = vec![0.0; 100];
        let mut output: Vec<f64> = vec![0.0; 8];
        b.iter(|| {
            simple.propagate(
                black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                &mut output,
                &mut storage1,
                &mut storage2,
            );
        })
    });
    let simple = SimpleNN::new(&[5, 8, 4, 1]);
    c.bench_function("feedforward 5-8-4-1 propagate", |b| {
        let mut storage1: Vec<f64> = vec![0.0; 100];
        let mut storage2: Vec<f64> = vec![0.0; 100];
        let mut output: Vec<f64> = vec![0.0; 8];
        b.iter(|| {
            simple.propagate(
                black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                &mut output,
                &mut storage1,
                &mut storage2,
            );
        })
    });
}

pub fn gru_benchmark(c: &mut Criterion) {
    let gru = GRUNetwork::new(&[8, 100, 100, 8]);
    c.bench_function("gru f64 8-100-100-8 propagate", |b| {
        let mut st = black_box(&gru).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });
    let gru = GRUNetwork::new(&[5, 8, 4, 1]);
    c.bench_function("gru f64 5-8-4-1 propagate", |b| {
        let mut st = black_box(&gru).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let gru = GRUNetworkF32::new(&[8, 100, 100, 8]);
    c.bench_function("gru f32 8-100-100-8 propagate", |b| {
        let mut st = black_box(&gru).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });
    let gru = GRUNetworkF32::new(&[5, 8, 4, 1]);
    c.bench_function("gru f32 5-8-4-1 propagate", |b| {
        let mut st = black_box(&gru).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });
}

pub fn lstm_benchmark(c: &mut Criterion) {
    let lstm = LSTMNetwork::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f64 8-100-100-8 propagate", |b| {
        let mut st = black_box(&lstm).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f32 8-100-100-8 propagate", |b| {
        let mut st = black_box(&lstm32).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f32 8-100-100-8 propagate (native f32)", |b| {
        let mut st = black_box(&lstm32).start();
        b.iter(|| {
            st.propagate32(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm = LSTMNetwork::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f64 5-8-4-1 propagate", |b| {
        let mut st = black_box(&lstm).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f32 5-8-4-1 propagate", |b| {
        let mut st = black_box(&lstm32).start();
        b.iter(|| {
            st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f32 5-8-4-1 propagate (native f32)", |b| {
        let mut st = black_box(&lstm32).start();
        b.iter(|| {
            st.propagate32(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0]));
        })
    });
}

criterion_group!(
    benches,
    feedforward_nn_benchmark,
    gru_benchmark,
    lstm_benchmark
);
criterion_main!(benches);
