use rug::Float;
use statrs::function::erf::erf;

fn series_is_trending_down_varpart(series: &[f64]) -> (f64, f64) {
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

    let mut xsum = 0.0;
    let mut ysum = 0.0;
    let mut xysum = 0.0;
    let mut x2sum = 0.0;

    for (idx, v) in series.iter().enumerate() {
        let v: f64 = *v;

        xsum += idx as f64;
        x2sum += (idx as f64) * (idx as f64);
        ysum += v;
        xysum += (idx as f64) * v;
    }

    let n: f64 = series.len() as f64;

    let btop = n * xysum - xsum * ysum;
    let bbottom = (n * x2sum) - xsum * xsum;
    let b = if btop == 0.0 && bbottom == 0.0 {
        1.0
    } else {
        btop / bbottom
    };
    let xavg = xsum / n;
    let yavg = ysum / n;
    let a = yavg - xavg * b;

    // compute variance
    let mut var: f64 = 0.0;
    for (idx, v) in series.iter().enumerate() {
        let v: f64 = *v;
        let idx: f64 = idx as f64;
        let prediction: f64 = a + b * idx;

        var += (v - prediction) * (v - prediction);
    }
    var = var / (series.len() as f64 - 2.0);

    // normal distribution parameters
    let m = b; // mean
    let var = (12.0 * var) / (n * n * n - n);

    (m, var)
}

pub fn series_is_trending_down(series: &[f64]) -> f64 {
    if series.len() < 3 {
        return 0.5;
    }

    let (m, var) = series_is_trending_down_varpart(series);

    // compute P(slope < 0) (where P(slope) ~ Gaussian(m, var))
    let p = 0.5 * (1.0 + erf((0.0 - m) / (var.sqrt() * (2.0_f64).sqrt())));
    p
}

pub fn series_is_trending_down_log(series: &[f64]) -> f64 {
    if series.len() < 3 {
        return (0.5_f64).ln();
    }

    let (m, var) = series_is_trending_down_varpart(series);
    // I couldn't figure out a proper way to use log(gaussian_cdf(x)) for very large/small values
    // of x so I am cheating. I'm using GMP's very high precision math instead and not using log
    // until the end.
    //
    // Works well enough for this purpose.
    let m: Float = Float::with_val(10000, m);
    let var: Float = Float::with_val(10000, var);
    let half: Float = Float::with_val(10000, 0.5);
    let one: Float = Float::with_val(10000, 1.0);
    let two: Float = Float::with_val(10000, 2.0);

    let p = (half * (one + ((-m) / (var.sqrt() * two.sqrt())).erf())).ln();
    p.to_f64()
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
            points.push(rng.gen_range(0.0, 4.0) + (x as f64 * 0.001));
        }
        let result2 = series_is_trending_down_log(&points);
        assert!(result2 < -10000.0);
    }
}
