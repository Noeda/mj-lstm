#[cfg(target_arch = "aarch64")]
pub use crate::simd_aarch64::*;
#[cfg(target_arch = "x86_64")]
pub use crate::simd_amd64::*;
use mj_autograd::*;
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct Vec4_F32 {
    pub(crate) v1: f32,
    pub(crate) v2: f32,
    pub(crate) v3: f32,
    pub(crate) v4: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct Vec4_F64 {
    pub(crate) v1: f64,
    pub(crate) v2: f64,
    pub(crate) v3: f64,
    pub(crate) v4: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct Vec8_F32 {
    pub(crate) v1: f32,
    pub(crate) v2: f32,
    pub(crate) v3: f32,
    pub(crate) v4: f32,
    pub(crate) v5: f32,
    pub(crate) v6: f32,
    pub(crate) v7: f32,
    pub(crate) v8: f32,
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

#[inline]
pub fn fast_sigmoid_derivative(x: f64) -> f64 {
    if x >= 0.0 {
        0.5 / (x + 1.0).powi(2)
    } else {
        0.5 / (-x + 1.0).powi(2)
    }
}

#[inline]
pub fn fast_tanh_derivative(x: f64) -> f64 {
    fast_sigmoid_derivative(x) * 2.0
}

#[inline]
pub fn fast_sigmoid_reverse(x: Reverse<f64>) -> Reverse<f64> {
    let x_abs = x.abs();
    Reverse::auto(0.5) + (x / (Reverse::auto(1.0) + x_abs)) * Reverse::auto(0.5)
}

#[inline]
pub fn fast_sigmoid_reverse32(x: Reverse<f32>) -> Reverse<f32> {
    let x_abs = x.abs();
    Reverse::auto(0.5) + (x / (Reverse::auto(1.0) + x_abs)) * Reverse::auto(0.5)
}

#[inline]
pub fn fast_sigmoid(x: f64) -> f64 {
    0.5 + (x / (1.0 + x.abs())) * 0.5
}

#[inline]
pub fn fast_tanh(x: f64) -> f64 {
    fast_sigmoid(x) * 2.0 - 1.0
}

#[inline]
pub fn inv_fast_sigmoid(y: f64) -> f64 {
    // x positive?
    if y >= 0.5 {
        1.0 / (1.0 / (2.0 * (y - 0.5)) - 1.0)
    } else {
        1.0 / (1.0 / (2.0 * (y - 0.5)) + 1.0)
    }
}

#[inline]
pub fn fast_relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[inline]
pub fn inv_fast_relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[inline]
pub fn fast_relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[inline]
pub fn inv_fast_tanh(y: f64) -> f64 {
    inv_fast_sigmoid((y + 1.0) / 2.0)
}

#[inline]
pub fn fast_silu(x: f64) -> f64 {
    x * fast_sigmoid(x)
}

#[inline]
pub fn fast_silu_derivative(x: f64) -> f64 {
    let s = fast_sigmoid(x);
    s + x * s * (1.0 - s)
}

#[inline]
pub fn fast_sigmoid32(x: f32) -> f32 {
    0.5 + (x / (1.0 + x.abs())) * 0.5
}

#[inline]
pub fn inv_sigmoid(x: f64) -> f64 {
    if x <= 0.0 {
        return -100_000.0;
    }
    if x >= 1.0 {
        return 100_000.0;
    }
    -(1.0 / x - 1.0).ln()
}

#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline]
pub fn inv_tanh(x: f64) -> f64 {
    if x <= -1.0 {
        return -100_000.0;
    }
    if x >= 1.0 {
        return 100_000.0;
    }
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

#[inline]
pub fn tanh_derivative(x: f64) -> f64 {
    let t = tanh(x);
    1.0 - t * t
}

#[inline]
pub fn softmax(vec: &mut [f64]) {
    if vec.len() == 0 {
        return;
    }
    let mut max_value: f64 = vec[0];
    for idx in 1..vec.len() {
        if vec[idx] > max_value {
            max_value = vec[idx];
        }
    }

    let mut denominator = 0.0;
    for value in vec.iter_mut() {
        let v = ((*value) - max_value).exp();
        denominator += v;
        *value = v;
    }

    for idx in 0..vec.len() {
        vec[idx] = vec[idx] / denominator;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn tanh_and_inv_tanh() {
        let mut rng = thread_rng();
        for _ in 0..1000 {
            let x = rng.gen_range(-10.0..10.0);
            let y = tanh(x);
            let x2 = inv_tanh(y);
            assert!((x - x2).abs() < 1e-6);
        }
    }
}
