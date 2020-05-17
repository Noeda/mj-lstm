use crossbeam::thread;
use mj_lstm::lstm::*;
use mj_lstm::rnn::*;
use std::sync::atomic::*;

fn bench(millis: u64) -> (std::time::Duration, usize) {
    let ones = vec![0.0; 50];
    let net = LSTMNetworkF32::new(&[50, 50, 50, 1]);

    let counter = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let start = AtomicBool::new(false);

    let total_sz = AtomicU64::new(0);

    thread::scope(|s| {
        let ncpus = num_cpus::get();
        let mut handles = vec![];
        for _ in 0..ncpus {
            handles.push(s.spawn(|_| {
                let mut st = net.start();
                while !start.load(Ordering::Relaxed) {}
                while !stop.load(Ordering::Relaxed) {
                    st.propagate32(&ones);
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                let out = st.propagate32(&ones);
                counter.fetch_add(1, Ordering::Relaxed);
                total_sz.fetch_add(out[0] as u64, Ordering::Relaxed);
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let start_time = std::time::Instant::now();
        start.store(true, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(millis));
        stop.store(true, Ordering::Relaxed);
        let count = counter.load(Ordering::Relaxed);
        let end_time = std::time::Instant::now();
        println!("Ensure propagate ran: {}", total_sz.load(Ordering::Relaxed));
        return (end_time - start_time, count);
    })
    .unwrap()
}

pub fn main() {
    // mul_add_scalar2 = 3
    // mul_add_scalar = 2
    // fast_sigmoid = 7
    let avx2_instructions_per_propagation = (50 * 50 * (3 + 2 + 7)) + (50 * 50 * (3 + 2 + 7));
    println!("warm-up");
    println!("{:?}", bench(3000));
    println!("benchmark");
    let (time_taken, propagations) = bench(10000);
    println!("Time taken: {:?}", time_taken);
    println!("Propagations: {}", propagations);
    println!(
        "AVX2 instructions: {}",
        avx2_instructions_per_propagation * propagations / 10 / 2
    );
    println!(
        "FLOPS: {}",
        avx2_instructions_per_propagation * propagations / 10 / 2 * 8
    );
}
