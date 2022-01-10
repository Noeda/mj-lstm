use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mj_lstm::gru::*;
use mj_lstm::lstm::*;
use mj_lstm::rnn::*;

pub fn gru_benchmark(c: &mut Criterion) {
    let gru = GRUNetwork::new(&[5, 8, 4, 1]);
    c.bench_function("GRUNetwork f64 5-8-4-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&gru).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        });
    });

    let gru = GRUNetworkSIMD::new(&[5, 8, 4, 1]);
    c.bench_function("GRUNetworkSIMD f64 5-8-4-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&gru).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        });
    });

    let gru = GRUNetwork::new(&[5, 80, 40, 1]);
    c.bench_function("GRUNetwork f64 5-80-40-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&gru).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        });
    });

    let gru = GRUNetworkSIMD::new(&[5, 80, 40, 1]);
    c.bench_function("GRUNetworkSIMD f64 5-80-40-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&gru).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        });
    });
}

pub fn lstm_benchmark(c: &mut Criterion) {
    let lstm = LSTMNetwork::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f64 8-100-100-8 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f32 8-100-100-8 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm32).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[8, 100, 100, 8]);
    c.bench_function("lstm f32 8-100-100-8 propagate (native f32)", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm32).start();
            black_box(st.propagate32(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])));
        })
    });

    let lstm = LSTMNetwork::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f64 5-8-4-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f32 5-8-4-1 propagate", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm32).start();
            black_box(st.propagate(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        })
    });

    let lstm32 = LSTMNetworkF32::new(&[5, 8, 4, 1]);
    c.bench_function("lstm f32 5-8-4-1 propagate (native f32)", |b| {
        b.iter(|| {
            let mut st = black_box(&lstm32).start();
            black_box(st.propagate32(black_box(&[0.0, 0.0, 0.0, 0.0, 0.0])));
        })
    });
}

criterion_group!(benches, gru_benchmark, lstm_benchmark);
criterion_main!(benches);
