use crate::gradient::GradientRecordF64;
#[cfg(target_arch = "aarch64")]
use crate::simd_aarch64::*;
#[cfg(target_arch = "x86_64")]
use crate::simd_amd64::*;

pub trait Unpackable {
    fn from_f64_vec(v: &[f64]) -> Vec<Self>
    where
        Self: Sized;
}

impl Unpackable for f64 {
    fn from_f64_vec(v: &[f64]) -> Vec<Self> {
        v.to_vec()
    }
}

impl Unpackable for GradientRecordF64 {
    fn from_f64_vec(v: &[f64]) -> Vec<Self> {
        let mut result_len = v.len() / 4;
        if v.len() % 4 > 0 {
            result_len += 1;
        }
        let mut result = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let x1 = v.get(i * 4).unwrap_or(&0.0);
            let x2 = v.get(i * 4 + 1).unwrap_or(&0.0);
            let x3 = v.get(i * 4 + 2).unwrap_or(&0.0);
            let x4 = v.get(i * 4 + 3).unwrap_or(&0.0);

            result.push(GradientRecordF64::new(*x1, *x2, *x3, *x4));
        }
        result
    }
}

impl Unpackable for F64x4 {
    fn from_f64_vec(v: &[f64]) -> Vec<Self> {
        let mut result_len = v.len() / 4;
        if v.len() % 4 > 0 {
            result_len += 1;
        }
        let mut result = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let x1 = v.get(i * 4).unwrap_or(&0.0);
            let x2 = v.get(i * 4 + 1).unwrap_or(&0.0);
            let x3 = v.get(i * 4 + 2).unwrap_or(&0.0);
            let x4 = v.get(i * 4 + 3).unwrap_or(&0.0);

            result.push(unsafe { F64x4::new(*x1, *x2, *x3, *x4) });
        }
        result
    }
}

impl Unpackable for F32x8 {
    fn from_f64_vec(v: &[f64]) -> Vec<Self> {
        let mut result_len = v.len() / 8;
        if v.len() % 8 > 0 {
            result_len += 1;
        }
        let mut result = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let x1 = v.get(i * 8).unwrap_or(&0.0);
            let x2 = v.get(i * 8 + 1).unwrap_or(&0.0);
            let x3 = v.get(i * 8 + 2).unwrap_or(&0.0);
            let x4 = v.get(i * 8 + 3).unwrap_or(&0.0);
            let x5 = v.get(i * 8 + 4).unwrap_or(&0.0);
            let x6 = v.get(i * 8 + 5).unwrap_or(&0.0);
            let x7 = v.get(i * 8 + 6).unwrap_or(&0.0);
            let x8 = v.get(i * 8 + 7).unwrap_or(&0.0);

            result.push(unsafe {
                F32x8::new(
                    *x1 as f32, *x2 as f32, *x3 as f32, *x4 as f32, *x5 as f32, *x6 as f32,
                    *x7 as f32, *x8 as f32,
                )
            });
        }
        result
    }
}

pub trait FromF64 {
    fn from_f64(f: f64) -> Self;
}
