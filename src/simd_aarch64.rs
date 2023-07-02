use crate::simd_common::*;
use core::arch::aarch64::*;

/* These types are really based on AVX2 types, I am just too lazy to change all the LSTM code so
 * we'll expose the same F64x4, F32x8 F32x4 etc. even though internally aarch64 does not have such
 * wide types. */

#[repr(C)]
#[derive(Copy, Clone)]
pub union F64x4 {
    pub(crate) val: (float64x2_t, float64x2_t),
    pub(crate) vec: Vec4_F64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union F32x8 {
    pub(crate) val: (float32x4_t, float32x4_t),
    pub(crate) vec: Vec8_F32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union F32x4 {
    pub(crate) val: float32x4_t,
    pub(crate) vec: Vec4_F32,
}

impl F32x4 {
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        F32x4 {
            vec: Vec4_F32 {
                v1: x1,
                v2: x2,
                v3: x3,
                v4: x4,
            },
        }
    }

    #[inline]
    pub(crate) unsafe fn mul_add_scalar(&mut self, other1: f32, other2: F32x4) {
        let broadcast_other1: float32x4_t = vmovq_n_f32(other1);
        self.val = vfmaq_f32(self.val, other2.val, broadcast_other1);
    }

    #[inline]
    pub(crate) unsafe fn mul_add_scalar2(&mut self, other1: f32, other2: f32, other3: F32x4) {
        let acc = F32x4::new(other1, other1, other2, other2);
        self.val = vfmaq_f32(self.val, other3.val, acc.val);
    }

    #[inline]
    pub(crate) fn zero(&mut self) {
        self.val = unsafe { vdupq_n_f32(0.0) };
    }

    pub(crate) unsafe fn fast_sigmoid(&mut self) {
        let half: float32x4_t = vmovq_n_f32(0.5);
        let one: float32x4_t = vmovq_n_f32(1.0);
        let negzero: float32x4_t = vmovq_n_f32(-0.0);

        // Convert to integer (need for bitwise operations)
        let negzero_i: uint32x4_t = vmvnq_u32(std::mem::transmute(negzero));
        let val_i: uint32x4_t = std::mem::transmute(self.val);

        let self_abs: float32x4_t =
            std::mem::transmute(vandq_u32(negzero_i, std::mem::transmute(val_i)));

        let plus_one = vaddq_f32(one, self_abs);

        let xdivided = vdivq_f32(self.val, plus_one);

        let multiplied = vfmaq_f32(half, xdivided, half);

        self.val = multiplied;
    }

    #[inline]
    pub(crate) fn v1(&self) -> f32 {
        unsafe { self.vec.v1 }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f32 {
        unsafe { self.vec.v2 }
    }

    #[inline]
    pub(crate) fn v3(&self) -> f32 {
        unsafe { self.vec.v3 }
    }

    #[inline]
    pub(crate) fn v4(&self) -> f32 {
        unsafe { self.vec.v4 }
    }
}

impl F32x8 {
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        x1: f32,
        x2: f32,
        x3: f32,
        x4: f32,
        x5: f32,
        x6: f32,
        x7: f32,
        x8: f32,
    ) -> Self {
        F32x8 {
            vec: Vec8_F32 {
                v1: x1,
                v2: x2,
                v3: x3,
                v4: x4,
                v5: x5,
                v6: x6,
                v7: x7,
                v8: x8,
            },
        }
    }

    #[inline]
    pub(crate) fn add(&mut self, other: F32x8) {
        unsafe {
            self.val.0 = vaddq_f32(self.val.0, other.val.0);
            self.val.1 = vaddq_f32(self.val.1, other.val.1);
        }
    }

    #[inline]
    pub(crate) unsafe fn mul_add_scalar(&mut self, other1: f32, other2: F32x8) {
        let broadcast_other1: float32x4_t = vmovq_n_f32(other1);
        self.val.0 = vfmaq_f32(self.val.0, other2.val.0, broadcast_other1);
        self.val.1 = vfmaq_f32(self.val.1, other2.val.1, broadcast_other1);
    }

    #[inline]
    pub(crate) unsafe fn mul_add_scalar2(&mut self, other1: f32, other2: f32, other3: F32x8) {
        let broadcast_other1: float32x4_t = vmovq_n_f32(other1);
        let broadcast_other2: float32x4_t = vmovq_n_f32(other2);
        self.val.0 = vfmaq_f32(self.val.0, other3.val.0, broadcast_other1);
        self.val.1 = vfmaq_f32(self.val.1, other3.val.1, broadcast_other2);
    }

    #[inline]
    pub(crate) fn zero(&mut self) {
        unsafe {
            self.val.0 = vdupq_n_f32(0.0);
            self.val.1 = vdupq_n_f32(0.0);
        }
    }

    #[inline]
    pub(crate) unsafe fn fast_sigmoid(&mut self) {
        let half: float32x4_t = vmovq_n_f32(0.5);
        let one: float32x4_t = vmovq_n_f32(1.0);
        let negzero: float32x4_t = vmovq_n_f32(-0.0);

        // Convert to integer (need for bitwise operations)
        let negzero_i: uint32x4_t = vmvnq_u32(std::mem::transmute(negzero));
        let val_i1: uint32x4_t = std::mem::transmute(self.val.0);
        let val_i2: uint32x4_t = std::mem::transmute(self.val.1);

        let self_abs1: float32x4_t =
            std::mem::transmute(vandq_u32(negzero_i, std::mem::transmute(val_i1)));
        let self_abs2: float32x4_t =
            std::mem::transmute(vandq_u32(negzero_i, std::mem::transmute(val_i2)));

        let plus_one1 = vaddq_f32(one, self_abs1);
        let plus_one2 = vaddq_f32(one, self_abs2);

        let xdivided1 = vdivq_f32(self.val.0, plus_one1);
        let xdivided2 = vdivq_f32(self.val.1, plus_one2);

        let multiplied1 = vfmaq_f32(half, xdivided1, half);
        let multiplied2 = vfmaq_f32(half, xdivided2, half);

        self.val.0 = multiplied1;
        self.val.1 = multiplied2;
    }

    #[inline]
    pub(crate) fn v1(&self) -> f32 {
        unsafe { self.vec.v1 }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f32 {
        unsafe { self.vec.v2 }
    }

    #[inline]
    pub(crate) fn v3(&self) -> f32 {
        unsafe { self.vec.v3 }
    }

    #[inline]
    pub(crate) fn v4(&self) -> f32 {
        unsafe { self.vec.v4 }
    }

    #[inline]
    pub(crate) fn v5(&self) -> f32 {
        unsafe { self.vec.v5 }
    }

    #[inline]
    pub(crate) fn v6(&self) -> f32 {
        unsafe { self.vec.v6 }
    }

    #[inline]
    pub(crate) fn v7(&self) -> f32 {
        unsafe { self.vec.v7 }
    }

    #[inline]
    pub(crate) fn v8(&self) -> f32 {
        unsafe { self.vec.v8 }
    }

    pub(crate) fn as_slice(&self) -> [f32; 8] {
        [
            self.v1(),
            self.v2(),
            self.v3(),
            self.v4(),
            self.v5(),
            self.v6(),
            self.v7(),
            self.v8(),
        ]
    }
}

impl F64x4 {
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(x1: f64, x2: f64, x3: f64, x4: f64) -> Self {
        F64x4 {
            vec: Vec4_F64 {
                v1: x1,
                v2: x2,
                v3: x3,
                v4: x4,
            },
        }
    }

    #[inline]
    pub(crate) fn broadcast(x: f64) -> Self {
        unsafe {
            let broadcast: float64x2_t = vmovq_n_f64(x);
            F64x4 {
                val: (broadcast, broadcast),
            }
        }
    }

    #[inline]
    pub(crate) fn add(&mut self, other: F64x4) {
        unsafe {
            self.val.0 = vaddq_f64(self.val.0, other.val.0);
            self.val.1 = vaddq_f64(self.val.1, other.val.1);
        }
    }

    #[inline]
    pub(crate) fn sub(&mut self, other: F64x4) {
        unsafe {
            self.val.0 = vsubq_f64(self.val.0, other.val.0);
            self.val.1 = vsubq_f64(self.val.1, other.val.1);
        }
    }

    #[inline]
    pub(crate) fn mul(&mut self, other: F64x4) {
        unsafe {
            self.val.0 = vmulq_f64(self.val.0, other.val.0);
            self.val.1 = vmulq_f64(self.val.1, other.val.1);
        }
    }

    #[inline]
    pub(crate) fn add_scalar(&mut self, other: f64) {
        unsafe {
            let broadcast_other: float64x2_t = vmovq_n_f64(other);
            self.val.0 = vaddq_f64(self.val.0, broadcast_other);
            self.val.1 = vaddq_f64(self.val.1, broadcast_other);
        }
    }

    #[inline]
    pub(crate) fn sqrt(&mut self) {
        unsafe {
            self.val.0 = vsqrtq_f64(self.val.0);
            self.val.1 = vsqrtq_f64(self.val.1);
        }
    }

    #[inline]
    pub(crate) fn div_scalar(&mut self, other: f64) {
        unsafe {
            let broadcast_other: float64x2_t = vmovq_n_f64(other);
            self.val.0 = vdivq_f64(self.val.0, broadcast_other);
            self.val.1 = vdivq_f64(self.val.1, broadcast_other);
        }
    }

    #[inline]
    pub(crate) fn div(&mut self, other: F64x4) {
        unsafe {
            self.val.0 = vdivq_f64(self.val.0, other.val.0);
            self.val.1 = vdivq_f64(self.val.1, other.val.1);
        }
    }

    #[inline]
    pub(crate) fn mul_add_scalar(&mut self, other1: f64, other2: F64x4) {
        unsafe {
            let broadcast_other1: float64x2_t = vmovq_n_f64(other1);
            self.val.0 = vfmaq_f64(self.val.0, other2.val.0, broadcast_other1);
            self.val.1 = vfmaq_f64(self.val.1, other2.val.1, broadcast_other1);
        }
    }

    #[inline]
    pub(crate) fn mul_add_scalar2(&mut self, other1: f64, other2: f64, other3: F64x4) {
        unsafe {
            let broadcast_other1: float64x2_t = vmovq_n_f64(other1);
            let broadcast_other2: float64x2_t = vmovq_n_f64(other2);
            self.val.0 = vfmaq_f64(self.val.0, other3.val.0, broadcast_other1);
            self.val.1 = vfmaq_f64(self.val.1, other3.val.1, broadcast_other2);
        }
    }

    #[inline]
    pub(crate) fn zero(&mut self) {
        unsafe {
            self.val.0 = vdupq_n_f64(0.0);
            self.val.1 = vdupq_n_f64(0.0);
        }
    }

    #[inline]
    pub(crate) fn fast_sigmoid(&mut self) {
        unsafe {
            let half: float64x2_t = vmovq_n_f64(0.5);
            let one: float64x2_t = vmovq_n_f64(1.0);
            let negzero: float64x2_t = vmovq_n_f64(-0.0);

            // Convert to integer (need for bitwise operations)
            let negzero_i: uint64x2_t =
                std::mem::transmute(vmvnq_u32(std::mem::transmute(negzero)));
            let val_i1: uint64x2_t = std::mem::transmute(self.val.0);
            let val_i2: uint64x2_t = std::mem::transmute(self.val.1);

            let self_abs1: float64x2_t =
                std::mem::transmute(vandq_u64(negzero_i, std::mem::transmute(val_i1)));
            let self_abs2: float64x2_t =
                std::mem::transmute(vandq_u64(negzero_i, std::mem::transmute(val_i2)));

            let plus_one1 = vaddq_f64(one, self_abs1);
            let plus_one2 = vaddq_f64(one, self_abs2);

            let xdivided1 = vdivq_f64(self.val.0, plus_one1);
            let xdivided2 = vdivq_f64(self.val.1, plus_one2);

            let multiplied1 = vfmaq_f64(half, xdivided1, half);
            let multiplied2 = vfmaq_f64(half, xdivided2, half);

            self.val.0 = multiplied1;
            self.val.1 = multiplied2;
        }
    }

    #[inline]
    pub(crate) fn sigmoid(&mut self) {
        unsafe {
            self.vec.v1 = sigmoid(self.vec.v1);
            self.vec.v2 = sigmoid(self.vec.v2);
            self.vec.v3 = sigmoid(self.vec.v3);
            self.vec.v4 = sigmoid(self.vec.v4);
        }
    }

    pub(crate) fn v1_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v1());
        }
        result
    }

    pub(crate) fn v2_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v2());
        }
        result
    }

    pub(crate) fn v3_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v3());
        }
        result
    }

    pub(crate) fn v4_vec(s: &[Self]) -> Vec<f64> {
        let mut result = Vec::with_capacity(s.len());
        for idx in 0..s.len() {
            result.push(s[idx].v4());
        }
        result
    }

    fn vec_from_vec64(x1: &[f64], x2: &[f64], x3: &[f64], x4: &[f64]) -> Vec<F64x4> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(F64x4::new(x1[idx], x2[idx], x3[idx], x4[idx]));
        }

        result
    }

    pub(crate) fn vecs_from_vecs64(
        x1: &[Vec<f64>],
        x2: &[Vec<f64>],
        x3: &[Vec<f64>],
        x4: &[Vec<f64>],
    ) -> Vec<Vec<F64x4>> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(F64x4::vec_from_vec64(
                &x1[idx], &x2[idx], &x3[idx], &x4[idx],
            ));
        }

        result
    }

    #[inline]
    pub(crate) fn v1(&self) -> f64 {
        unsafe { self.vec.v1 }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f64 {
        unsafe { self.vec.v2 }
    }

    #[inline]
    pub(crate) fn v3(&self) -> f64 {
        unsafe { self.vec.v3 }
    }

    #[inline]
    pub(crate) fn v4(&self) -> f64 {
        unsafe { self.vec.v4 }
    }

    pub(crate) fn as_slice(&self) -> [f64; 4] {
        [self.v1(), self.v2(), self.v3(), self.v4()]
    }
}
