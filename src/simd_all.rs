use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(target_arch = "aarch64")]
pub(crate) use crate::simd_aarch64::*;
#[cfg(target_arch = "x86_64")]
pub(crate) use crate::simd_amd64::*;
pub(crate) use crate::simd_common::*;

impl PartialOrd for F32x8 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        unsafe { Vec8_F32::from(*self).partial_cmp(&Vec8_F32::from(*other)) }
    }
}

impl PartialOrd for F64x4 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        unsafe { Vec4_F64::from(*self).partial_cmp(&Vec4_F64::from(*other)) }
    }
}

impl PartialOrd for F64x2 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        unsafe { Vec2_F64::from(*self).partial_cmp(&Vec2_F64::from(*other)) }
    }
}

impl PartialEq for F32x8 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { Vec8_F32::from(*self).eq(&Vec8_F32::from(*other)) }
    }
}

impl PartialEq for F64x4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { Vec4_F64::from(*self).eq(&Vec4_F64::from(*other)) }
    }
}

impl PartialEq for F64x2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { Vec2_F64::from(*self).eq(&Vec2_F64::from(*other)) }
    }
}

impl Serialize for F32x8 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unsafe { Vec8_F32::from(*self).serialize(serializer) }
    }
}

impl Serialize for F64x4 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unsafe { Vec4_F64::from(*self).serialize(serializer) }
    }
}

impl Serialize for F64x2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unsafe { Vec2_F64::from(*self).serialize(serializer) }
    }
}

impl<'de> Deserialize<'de> for F32x8 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec8 = Vec8_F32::deserialize(deserializer)?;
        Ok(unsafe {
            F32x8::new(
                vec8.v1, vec8.v2, vec8.v3, vec8.v4, vec8.v5, vec8.v6, vec8.v7, vec8.v8,
            )
        })
    }
}

impl<'de> Deserialize<'de> for F64x4 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec4 = Vec4_F64::deserialize(deserializer)?;
        Ok(unsafe { F64x4::new(vec4.v1, vec4.v2, vec4.v3, vec4.v4) })
    }
}

impl<'de> Deserialize<'de> for F64x2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec2 = Vec2_F64::deserialize(deserializer)?;
        Ok(unsafe { F64x2::new(vec2.v1, vec2.v2) })
    }
}

impl fmt::Debug for F64x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            write!(
                f,
                "F64x4 {{ v1: {}, v2: {}, v3: {}, v4: {} }}",
                self.v1(),
                self.v2(),
                self.v3(),
                self.v4()
            )
        }
    }
}

impl fmt::Debug for F64x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe { write!(f, "F64x4 {{ v1: {}, v2: {} }}", self.v1(), self.v2(),) }
    }
}

impl fmt::Debug for F32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            write!(
                f,
                "F32x8 {{ v1: {}, v2: {}, v3: {}, v4: {}, v5: {}, v6: {}, v7: {}, v8: {} }}",
                self.v1(),
                self.v2(),
                self.v3(),
                self.v4(),
                self.v5(),
                self.v6(),
                self.v7(),
                self.v8()
            )
        }
    }
}

pub(crate) fn vecf64_to_vecf32(v: &[F64x4]) -> Vec<F32x8> {
    let mut sz = v.len() / 2;
    if v.len() % 2 == 1 {
        sz += 1;
    }
    let mut result = Vec::with_capacity(sz);
    unsafe {
        let zero: F64x4 = F64x4::new(0.0, 0.0, 0.0, 0.0);
        for i in 0..sz {
            let p1 = v.get(i * 2).unwrap_or(&zero);
            let p2 = v.get(i * 2 + 1).unwrap_or(&zero);
            result.push(F32x8::new(
                p1.v1() as f32,
                p1.v2() as f32,
                p1.v3() as f32,
                p1.v4() as f32,
                p2.v1() as f32,
                p2.v2() as f32,
                p2.v3() as f32,
                p2.v4() as f32,
            ));
        }
    }
    result
}

pub(crate) fn vecf32_to_vecf64(v: &[F32x8]) -> Vec<F64x4> {
    let sz = v.len() * 2;
    let mut result = Vec::with_capacity(sz);
    for i in 0..v.len() {
        unsafe {
            result.push(F64x4::new(
                v[i].v1() as f64,
                v[i].v2() as f64,
                v[i].v3() as f64,
                v[i].v4() as f64,
            ));
            result.push(F64x4::new(
                v[i].v5() as f64,
                v[i].v6() as f64,
                v[i].v7() as f64,
                v[i].v8() as f64,
            ));
        }
    }
    result
}

impl From<Vec4_F32> for F32x4 {
    fn from(thing: Vec4_F32) -> Self {
        unsafe { F32x4::new(thing.v1, thing.v2, thing.v3, thing.v4) }
    }
}

impl From<F32x4> for Vec4_F32 {
    fn from(thing: F32x4) -> Self {
        Vec4_F32 {
            v1: thing.v1(),
            v2: thing.v2(),
            v3: thing.v3(),
            v4: thing.v4(),
        }
    }
}

impl From<Vec8_F32> for F32x8 {
    fn from(thing: Vec8_F32) -> Self {
        unsafe {
            F32x8::new(
                thing.v1, thing.v2, thing.v3, thing.v4, thing.v5, thing.v6, thing.v7, thing.v8,
            )
        }
    }
}

impl From<F32x8> for Vec8_F32 {
    fn from(thing: F32x8) -> Self {
        Vec8_F32 {
            v1: thing.v1(),
            v2: thing.v2(),
            v3: thing.v3(),
            v4: thing.v4(),
            v5: thing.v5(),
            v6: thing.v6(),
            v7: thing.v7(),
            v8: thing.v8(),
        }
    }
}

impl From<Vec4_F64> for F64x4 {
    fn from(thing: Vec4_F64) -> Self {
        unsafe { F64x4::new(thing.v1, thing.v2, thing.v3, thing.v4) }
    }
}

impl From<F64x4> for Vec4_F64 {
    fn from(thing: F64x4) -> Self {
        Vec4_F64 {
            v1: thing.v1(),
            v2: thing.v2(),
            v3: thing.v3(),
            v4: thing.v4(),
        }
    }
}

impl From<Vec2_F64> for F64x2 {
    fn from(thing: Vec2_F64) -> Self {
        unsafe { F64x2::new(thing.v1, thing.v2) }
    }
}

impl From<F64x2> for Vec2_F64 {
    fn from(thing: F64x2) -> Self {
        Vec2_F64 {
            v1: thing.v1(),
            v2: thing.v2(),
        }
    }
}
