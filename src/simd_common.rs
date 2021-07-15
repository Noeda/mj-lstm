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
pub fn fast_sigmoid(x: f64) -> f64 {
    0.5 + (x / (1.0 + x.abs())) * 0.5
}

#[inline]
pub fn fast_sigmoid32(x: f32) -> f32 {
    0.5 + (x / (1.0 + x.abs())) * 0.5
}

pub fn inv_sigmoid(x: f64) -> f64 {
    if x <= 0.0 {
        return -100_000.0;
    }
    if x >= 1.0 {
        return 100_000.0;
    }
    -(1.0 / x - 1.0).ln()
}
