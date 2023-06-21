use mj_lstm::lstm::*;
use mj_lstm::meta_lstm::*;
use mj_lstm::series_is_trending_down_log;
use rand::{thread_rng, Rng, SeedableRng};
use rayon::prelude::*;
use rcmaes::cosyne::*;
use std::collections::VecDeque;

pub fn main() {
    let mut lstm: MetaNN = MetaNN::new(4);
    let mut cma = Cosyne::new(
        &lstm,
        &CosyneSettings::default()
            .sigma(0.01)
            .shrinkage_multiplier(0.98),
    );
    let mut epochs: usize = 0;

    let mut rng = thread_rng();
    let mut scores_over_time: VecDeque<f64> = VecDeque::with_capacity(1000);
    loop {
        let seed: u64 = rng.gen();

        let candidates = cma.ask();
        let candidates: Vec<CosyneCandidate<MetaNN>> = candidates
            .into_par_iter()
            .map(|mut candidate| {
                let score = evaluate(candidate.item(), seed);
                candidate.set_score(score);
                candidate
            })
            .collect();

        let mut best_score: f64 = std::f64::INFINITY;
        let mut best_model = candidates[0].item().clone();
        for cand in candidates.iter() {
            if cand.score() < best_score {
                best_score = cand.score();
                best_model = cand.item().clone();
            }
        }
        println!("epoch={} sigma={} {:?}", epochs, cma.sigma(), best_score);
        println!("---");
        scores_over_time.push_back(best_score);
        while scores_over_time.len() > 1000 {
            scores_over_time.pop_front();
        }
        let scores: Vec<f64> = scores_over_time.iter().map(|x| *x).collect();
        let p = series_is_trending_down_log(&scores);
        println!("Going down change: {:.5}", p.exp());
        demonstrate(&best_model);
        cma.tell(candidates);
        epochs += 1;
        if epochs % 1000 == 0 {
            if p.exp() <= 0.5 {
                let old_sigma = cma.sigma();
                cma.set_sigma(old_sigma * 0.5);
                println!("Sigma adjusted: {:.5} -> {:.5}", old_sigma, cma.sigma());
            }
        }
    }
}

fn demonstrate(model: &MetaNN) {
    let mut rng = thread_rng();
    for _ in 0..5 {
        let mut lstm_state = MetaNNState::new(5, &[10, 10, 10, 10], 1, model.clone());

        for _ in 0..50 {
            let mut input: Vec<f64> = vec![0.0; 5];
            let mut answer: f64 = 0.0;
            for i in 0..5 {
                let x: f64 = rng.gen_range(0.0, 1.0);
                input[i] = x;
                answer += x;
            }
            answer /= input.len() as f64;

            let mut out: Vec<f64> = vec![0.0];
            lstm_state.propagate(&input, &mut out);
            let output_before_training = out[0];
            lstm_state.set_errors(&[answer - output_before_training]);
        }

        let mut input: Vec<f64> = vec![0.0; 5];
        let mut answer: f64 = 0.0;
        for i in 0..5 {
            let x: f64 = rng.gen_range(0.0, 1.0);
            input[i] = x;
            if x > answer {
                answer = x;
            }
        }

        let mut out: Vec<f64> = vec![0.0];
        lstm_state.propagate(&input, &mut out);
        let output_after_training = out[0];
        println!("answer={:.3}, after={:.3}", answer, output_after_training);
    }
}

fn evaluate(model: &MetaNN, seed: u64) -> f64 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut predictions: Vec<f64> = Vec::with_capacity(5);

    let mut score: f64 = 0.0;
    let mut train_epochs: usize = rng.gen_range(1, 50);
    for _ in 0..50 {
        let mut lstm_state = MetaNNState::new(5, &[10, 10, 10, 10], 1, model.clone());
        let mut prev_prediction: f64 = -100.0;
        for _ in 0..train_epochs {
            let mut input: Vec<f64> = vec![0.0; 5];
            let mut answer: f64 = 0.0;
            for i in 0..5 {
                let x: f64 = rng.gen_range(0.0, 1.0);
                input[i] = x;
                if x > answer {
                    answer = x;
                }
            }

            let mut out: Vec<f64> = vec![0.0];
            lstm_state.propagate(&input, &mut out);
            let output_before_training = out[0];
            let score_before_training = (answer - output_before_training).abs();
            lstm_state.set_errors(&[answer - output_before_training]);
            score += (answer - output_before_training).abs();

            predictions.push(output_before_training);
        }
    }

    let mut smallest_diff: f64 = std::f64::INFINITY;
    let mut prev: f64 = 0.0;
    for pred in predictions.iter() {
        let diff = (pred - prev).abs();
        if diff < smallest_diff {
            smallest_diff = diff;
        }
        prev = *pred;
    }
    /*
    if smallest_diff < 0.001 {
        score += 1000.0;
    }
    */
    //println!("{:?} {:?}", predictions, smallest_diff);
    score
}
