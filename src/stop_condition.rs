use rug::Float;
use statrs::function::erf::erf;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

fn series_is_trending_down_varpart<T, F>(f64_to_t: F, series: &[f64]) -> (T, T)
where
    T: Clone
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Sub<Output = T>
        + Add<Output = T>
        + PartialEq,
    F: Fn(f64) -> T,
{
    assert!(series.len() >= 3);

    // 1. Fit Ordinary Least Squares to the series
    // 2. Compute variance of it
    // 3. Compute probability that the series is trending up

    // Step 1. using Wikipedia algorithm
    //
    // y = a + b*x + e
    //
    // Where 'a' is base, 'b' is the slope, and 'e' is noise
    //
    // 'series' gives y-coordinates and x-coordinates are considered to be just 0, 1, 2, 3, etc.

    let zero = f64_to_t(0.0);

    let mut xsum: T = zero.clone();
    let mut ysum: T = zero.clone();
    let mut xysum: T = zero.clone();
    let mut x2sum: T = zero.clone();

    for (idx, v) in series.iter().enumerate() {
        let v: T = f64_to_t(*v);
        let tidx: T = f64_to_t(idx as f64);

        xsum += tidx.clone();
        x2sum += tidx.clone() * tidx.clone();
        ysum += v.clone();
        xysum += tidx * v;
    }

    let n: T = f64_to_t(series.len() as f64);

    let btop: T = n.clone() * xysum - xsum.clone() * ysum.clone();
    let bbottom: T = (n.clone() * x2sum) - xsum.clone() * xsum.clone();
    let b: T = if btop == zero && bbottom == zero {
        zero + f64_to_t(1.0)
    } else {
        btop / bbottom
    };
    let xavg = xsum / n.clone();
    let yavg = ysum / n.clone();
    let a = yavg - xavg * b.clone();

    // compute variance
    let mut var: T = f64_to_t(0.0);
    for (idx, v) in series.iter().enumerate() {
        let v: T = f64_to_t(*v);
        let idx: T = f64_to_t(idx as f64);
        let prediction: T = a.clone() + b.clone() * idx;

        var += (v.clone() - prediction.clone()) * (v - prediction);
    }
    var = var / f64_to_t(series.len() as f64 - 2.0);

    // normal distribution parameters
    let m: T = b; // mean
    let var = (f64_to_t(12.0) * var) / (n.clone() * n.clone() * n.clone() - n);

    (m, var)
}

pub fn series_is_trending_down(series: &[f64]) -> f64 {
    if series.len() < 3 {
        return 0.5;
    }

    let (m, var) = series_is_trending_down_varpart(|x| x, series);

    // compute P(slope < 0) (where P(slope) ~ Gaussian(m, var))
    let p = 0.5 * (1.0 + erf((0.0 - m) / (var.sqrt() * (2.0_f64).sqrt())));
    p
}

fn down_log_approximation(x: f64) -> f64 {
    let x = -x;

    let model1 = 4.140209418032632;
    let model2 = 0.06251226694149614;
    let model3 = 1.0000186632995702;
    let model4 = -0.018755001842882372;
    let model5 = 1.1170972713775695;
    let model6 = 0.7985471585333958;
    let model7 = 1.0020395787570657;
    let model8 = -0.37493131816552866;

    if x >= 70.0 {
        return -(model1 + x * model2 + x * x * model3 + (x.powf(1.2) * model4));
    } else {
        return -(model5 + x * model6 + x * x * model7 + (x.powf(1.2) * model8));
    }
}

pub fn series_is_trending_down_log(series: &[f64]) -> f64 {
    if series.len() < 3 {
        return (0.5_f64).ln();
    }

    let (fm, fvar) = series_is_trending_down_varpart(|x| x, series);

    let (m, var) = {
        if fvar == 0.0 || fvar != fvar || fvar.is_infinite() {
            series_is_trending_down_varpart(|x| Float::with_val(10000, x), series)
        } else {
            (Float::with_val(10000, fm), Float::with_val(10000, fvar))
        }
    };
    // I couldn't figure out a proper way to use log(gaussian_cdf(x)) for very large/small values
    // of x so I am cheating. I'm using GMP's very high precision math instead and not using log
    // until the end.
    //
    // Works well enough for this purpose.
    let half: Float = Float::with_val(10000, 0.5);
    let one: Float = Float::with_val(10000, 1.0);
    let two: Float = Float::with_val(10000, 2.0);

    let int = (-m.clone()) / (var.clone().sqrt() * two.clone().sqrt());

    let p = (half * (one + ((-m) / (var.sqrt() * two.sqrt())).erf())).ln();
    if int > -2.0 {
        p.to_f64()
    } else {
        down_log_approximation(int.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn simple_up_line() {
        let mut rng = thread_rng();
        let mut points = vec![];
        for x in 0..4000 {
            points.push(rng.gen_range(0.0, 4.0) + (x as f64) * 0.01);
        }
        let result = series_is_trending_down(&points);
        assert!(result < 0.00001);
        assert!(result >= 0.0);
    }

    #[test]
    fn simple_down_line() {
        let mut rng = thread_rng();
        let mut points = vec![];
        for x in 0..4000 {
            points.push(rng.gen_range(0.0, 4.0) + (x as f64) * -0.01);
        }
        let result = series_is_trending_down(&points);
        assert!(result > 0.99999);
        assert!(result <= 1.0);
    }

    #[test]
    fn straight_middle() {
        let mut rng = thread_rng();
        let mut points = vec![];
        for x in 0..4000 {
            points.push(rng.gen_range(0.0, 4.0));
        }
        let result = series_is_trending_down(&points);
        assert!(result > 0.1);
        assert!(result < 0.9);
    }

    #[test]
    fn log_and_normal_match_for_small_values() {
        let mut rng = thread_rng();
        let mut points = vec![];
        for x in 0..4000 {
            points.push(rng.gen_range(0.0, 4.0));
        }
        let result = series_is_trending_down(&points);
        let result2 = series_is_trending_down_log(&points).exp();
        assert!(result - result2 <= 0.01);
    }

    #[test]
    fn log_matches_for_extremely_unlikely_series() {
        let mut rng = thread_rng();
        let mut points = vec![];
        for x in 0..4000 {
            points.push(rng.gen_range(0.0, 4.0) + (x as f64 * 0.1));
        }
        let result2 = series_is_trending_down_log(&points);
        assert!(result2 < -10000.0);
    }
}
