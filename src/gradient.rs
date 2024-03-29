/*
 * Stuff to support gradient computing.
 */

// Type that's analogous to the SIMD types but uses gradient types.
// (Does actually use SIMD rip)

#[cfg(target_arch = "aarch64")]
pub(crate) use crate::simd_aarch64::*;
#[cfg(target_arch = "x86_64")]
pub(crate) use crate::simd_amd64::*;
use crate::simd_common::{fast_sigmoid_reverse, fast_sigmoid_reverse32};
use mj_autograd::*;

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct GradientRecordF64 {
    v1: Reverse<f64>,
    v2: Reverse<f64>,
    v3: Reverse<f64>,
    v4: Reverse<f64>,
}

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct GradientRecordF32 {
    v1: Reverse<f32>,
    v2: Reverse<f32>,
    v3: Reverse<f32>,
    v4: Reverse<f32>,
}

impl From<F64x4> for GradientRecordF64 {
    fn from(x: F64x4) -> Self {
        GradientRecordF64 {
            v1: Reverse::auto(x.v1()),
            v2: Reverse::auto(x.v2()),
            v3: Reverse::auto(x.v3()),
            v4: Reverse::auto(x.v4()),
        }
    }
}

impl From<F64x4> for GradientRecordF32 {
    fn from(x: F64x4) -> Self {
        GradientRecordF32 {
            v1: Reverse::auto(x.v1() as f32),
            v2: Reverse::auto(x.v2() as f32),
            v3: Reverse::auto(x.v3() as f32),
            v4: Reverse::auto(x.v4() as f32),
        }
    }
}

impl GradientRecordF64 {
    pub fn from_f64x4(x: F64x4, tape: Tape<f64>) -> Self {
        GradientRecordF64 {
            v1: Reverse::reversible(x.v1(), tape.clone()),
            v2: Reverse::reversible(x.v2(), tape.clone()),
            v3: Reverse::reversible(x.v3(), tape.clone()),
            v4: Reverse::reversible(x.v4(), tape),
        }
    }

    #[inline]
    pub fn new(v1: f64, v2: f64, v3: f64, v4: f64) -> Self {
        Self {
            v1: Reverse::auto(v1),
            v2: Reverse::auto(v2),
            v3: Reverse::auto(v3),
            v4: Reverse::auto(v4),
        }
    }

    #[inline]
    pub fn v1(&self) -> Reverse<f64> {
        self.v1.clone()
    }

    #[inline]
    pub fn v2(&self) -> Reverse<f64> {
        self.v2.clone()
    }

    #[inline]
    pub fn v3(&self) -> Reverse<f64> {
        self.v3.clone()
    }

    #[inline]
    pub fn v4(&self) -> Reverse<f64> {
        self.v4.clone()
    }

    #[inline]
    pub fn v1_mut(&mut self) -> &mut Reverse<f64> {
        &mut self.v1
    }

    #[inline]
    pub fn v2_mut(&mut self) -> &mut Reverse<f64> {
        &mut self.v2
    }

    #[inline]
    pub fn v3_mut(&mut self) -> &mut Reverse<f64> {
        &mut self.v3
    }

    #[inline]
    pub fn v4_mut(&mut self) -> &mut Reverse<f64> {
        &mut self.v4
    }

    #[inline]
    pub fn mul_add_scalar(&mut self, a: Reverse<f64>, b: &GradientRecordF64) {
        self.v1 = self.v1.clone() + a.clone() * b.v1.clone();
        self.v2 = self.v2.clone() + a.clone() * b.v2.clone();
        self.v3 = self.v3.clone() + a.clone() * b.v3.clone();
        self.v4 = self.v4.clone() + a * b.v4.clone();
    }

    #[inline]
    pub fn fast_sigmoid(&mut self) {
        self.v1 = fast_sigmoid_reverse(self.v1.clone());
        self.v2 = fast_sigmoid_reverse(self.v2.clone());
        self.v3 = fast_sigmoid_reverse(self.v3.clone());
        self.v4 = fast_sigmoid_reverse(self.v4.clone());
    }
}

impl GradientRecordF32 {
    pub fn from_f64x4(x: F64x4, tape: Tape<f32>) -> Self {
        GradientRecordF32 {
            v1: Reverse::reversible(x.v1() as f32, tape.clone()),
            v2: Reverse::reversible(x.v2() as f32, tape.clone()),
            v3: Reverse::reversible(x.v3() as f32, tape.clone()),
            v4: Reverse::reversible(x.v4() as f32, tape),
        }
    }

    #[inline]
    pub fn new(v1: f32, v2: f32, v3: f32, v4: f32) -> Self {
        Self {
            v1: Reverse::auto(v1),
            v2: Reverse::auto(v2),
            v3: Reverse::auto(v3),
            v4: Reverse::auto(v4),
        }
    }

    #[inline]
    pub fn v1(&self) -> Reverse<f32> {
        self.v1.clone()
    }

    #[inline]
    pub fn v2(&self) -> Reverse<f32> {
        self.v2.clone()
    }

    #[inline]
    pub fn v3(&self) -> Reverse<f32> {
        self.v3.clone()
    }

    #[inline]
    pub fn v4(&self) -> Reverse<f32> {
        self.v4.clone()
    }

    #[inline]
    pub fn v1_mut(&mut self) -> &mut Reverse<f32> {
        &mut self.v1
    }

    #[inline]
    pub fn v2_mut(&mut self) -> &mut Reverse<f32> {
        &mut self.v2
    }

    #[inline]
    pub fn v3_mut(&mut self) -> &mut Reverse<f32> {
        &mut self.v3
    }

    #[inline]
    pub fn v4_mut(&mut self) -> &mut Reverse<f32> {
        &mut self.v4
    }

    #[inline]
    pub fn mul_add_scalar(&mut self, a: Reverse<f32>, b: &GradientRecordF32) {
        self.v1 = self.v1.clone() + a.clone() * b.v1.clone();
        self.v2 = self.v2.clone() + a.clone() * b.v2.clone();
        self.v3 = self.v3.clone() + a.clone() * b.v3.clone();
        self.v4 = self.v4.clone() + a * b.v4.clone();
    }

    #[inline]
    pub fn fast_sigmoid(&mut self) {
        self.v1 = fast_sigmoid_reverse32(self.v1.clone());
        self.v2 = fast_sigmoid_reverse32(self.v2.clone());
        self.v3 = fast_sigmoid_reverse32(self.v3.clone());
        self.v4 = fast_sigmoid_reverse32(self.v4.clone());
    }
}
