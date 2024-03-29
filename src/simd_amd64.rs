use crate::simd_common::*;
use core::arch::x86_64::*;
use std::mem::MaybeUninit;

#[repr(C)]
#[derive(Copy, Clone)]
pub union F64x4 {
    pub(crate) val: __m256d,
    pub(crate) vec: Vec4_F64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union F32x8 {
    pub(crate) val: __m256,
    pub(crate) vec: Vec8_F32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union F32x4 {
    pub(crate) val: __m128,
    pub(crate) vec: Vec4_F32,
}

impl F32x4 {
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
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar(&mut self, other1: f32, other2: F32x4) {
        let broadcast_other1: __m128 = _mm_broadcast_ss(&other1);
        self.val = _mm_fmadd_ps(broadcast_other1, other2.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar2(&mut self, other1: f32, other2: f32, other3: F32x4) {
        let b_vec = F32x4::new(other1, other1, other2, other2);
        self.val = _mm_fmadd_ps(b_vec.val, other3.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn fast_sigmoid(&mut self) {
        let half = _mm_broadcast_ss(&0.5);
        let one = _mm_broadcast_ss(&1.0);
        let negzero = _mm_broadcast_ss(&-0.0);
        let self_abs = _mm_andnot_ps(negzero, self.val);
        let plus_one = _mm_add_ps(one, self_abs);
        let xdivided = _mm_div_ps(self.val, plus_one);
        self.val = _mm_fmadd_ps(xdivided, half, half)
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
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
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
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar(&mut self, other1: f32, other2: F32x8) {
        let broadcast_other1: __m256 = _mm256_broadcast_ss(&other1);
        self.val = _mm256_fmadd_ps(broadcast_other1, other2.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar2(&mut self, other1: f32, other2: f32, other3: F32x8) {
        let mut v: MaybeUninit<__m256> = MaybeUninit::<__m256>::uninit();
        let b: *mut __m256 = v.as_mut_ptr();
        let b1: *mut __m128 = b as *mut __m128;
        let b2: *mut __m128 = b1.add(1);
        *b1 = _mm_broadcast_ss(&other1);
        *b2 = _mm_broadcast_ss(&other2);
        self.val = _mm256_fmadd_ps(*b, other3.val, self.val);
    }

    #[inline]
    pub(crate) fn zero(&mut self) {
        unsafe { self.zero_shadow() }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn zero_shadow(&mut self) {
        self.val = _mm256_setzero_ps();
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn fast_sigmoid(&mut self) {
        let half = _mm256_broadcast_ss(&0.5);
        let one = _mm256_broadcast_ss(&1.0);
        let negzero = _mm256_broadcast_ss(&-0.0);
        let self_abs = _mm256_andnot_ps(negzero, self.val);
        let plus_one = _mm256_add_ps(one, self_abs);
        let xdivided = _mm256_div_ps(self.val, plus_one);
        self.val = _mm256_fmadd_ps(xdivided, half, half)
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
}

impl F64x4 {
    #[inline]
    pub(crate) fn new(x1: f64, x2: f64, x3: f64, x4: f64) -> Self {
        unsafe { Self::new_shadow(x1, x2, x3, x4) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn new_shadow(x1: f64, x2: f64, x3: f64, x4: f64) -> Self {
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
    pub(crate) fn broadcast(x1: f64) -> Self {
        unsafe { Self::broadcast_shadow(x1) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn broadcast_shadow(x1: f64) -> Self {
        F64x4 {
            vec: Vec4_F64 {
                v1: x1,
                v2: x1,
                v3: x1,
                v4: x1,
            },
        }
    }

    #[inline]
    pub(crate) fn nans_to_zero(&mut self) {
        unsafe { self.nans_to_zero_shadow() }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn nans_to_zero_shadow(&mut self) {
        let zero = _mm256_setzero_pd();
        let nan = _mm256_cmp_pd(self.val, self.val, _CMP_UNORD_Q);
        self.val = _mm256_blendv_pd(zero, self.val, nan);
    }

    #[inline]
    pub(crate) fn max_scalar(&mut self, scalar: f64, other: F64x4) {
        unsafe { self.max_scalar_shadow(scalar, other) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn max_scalar_shadow(&mut self, scalar: f64, other: F64x4) {
        let broadcast_scalar = _mm256_broadcast_sd(&scalar);
        self.val = _mm256_max_pd(broadcast_scalar, other.val);
    }

    #[inline]
    pub(crate) fn min_scalar(&mut self, scalar: f64, other: F64x4) {
        unsafe { self.min_scalar_shadow(scalar, other) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn min_scalar_shadow(&mut self, scalar: f64, other: F64x4) {
        let broadcast_scalar = _mm256_broadcast_sd(&scalar);
        self.val = _mm256_min_pd(broadcast_scalar, other.val);
    }

    #[inline]
    pub(crate) fn fast_sigmoid(&mut self) {
        unsafe { self.fast_sigmoid_shadow() }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn fast_sigmoid_shadow(&mut self) {
        let half = _mm256_broadcast_sd(&0.5);
        let one = _mm256_broadcast_sd(&1.0);
        let negzero = _mm256_broadcast_sd(&-0.0);
        let self_abs = _mm256_andnot_pd(negzero, self.val);
        let plus_one = _mm256_add_pd(one, self_abs);
        let xdivided = _mm256_div_pd(self.val, plus_one);
        self.val = _mm256_fmadd_pd(xdivided, half, half)
    }

    #[inline]
    pub(crate) fn zero(&mut self) {
        unsafe { self.zero_shadow() }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn zero_shadow(&mut self) {
        self.val = _mm256_setzero_pd();
    }

    #[inline]
    pub(crate) fn sub(&mut self, other: F64x4) {
        unsafe { self.sub_shadow(other) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_shadow(&mut self, other: F64x4) {
        self.val = _mm256_sub_pd(self.val, other.val);
    }

    #[inline]
    pub(crate) fn sqrt(&mut self) {
        unsafe { self.sqrt_shadow() }
    }

    #[inline]
    unsafe fn sqrt_shadow(&mut self) {
        self.val = _mm256_sqrt_pd(self.val);
    }

    #[inline]
    pub(crate) fn add(&mut self, other: F64x4) {
        unsafe { self.add_shadow(other) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_shadow(&mut self, other: F64x4) {
        self.val = _mm256_add_pd(other.val, self.val);
    }

    #[inline]
    pub(crate) fn add_scalar(&mut self, scalar: f64) {
        unsafe { self.add_scalar_shadow(scalar) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_shadow(&mut self, scalar: f64) {
        let scalar = _mm256_broadcast_sd(&scalar);
        self.val = _mm256_add_pd(scalar, self.val);
    }

    #[inline]
    pub(crate) fn div(&mut self, other: F64x4) {
        unsafe { self.div_shadow(other) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn div_shadow(&mut self, other: F64x4) {
        self.val = _mm256_div_pd(self.val, other.val);
    }

    #[inline]
    pub(crate) fn div_scalar(&mut self, scalar: f64) {
        unsafe { self.div_scalar_shadow(scalar) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn div_scalar_shadow(&mut self, scalar: f64) {
        let scalar = _mm256_broadcast_sd(&scalar);
        self.val = _mm256_div_pd(self.val, scalar);
    }

    #[inline]
    pub(crate) fn mul_add(&mut self, other1: F64x4, other: F64x4) {
        unsafe { self.mul_add_shadow(other1, other) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_add_shadow(&mut self, other1: F64x4, other2: F64x4) {
        self.val = _mm256_fmadd_pd(other1.val, other2.val, self.val);
    }

    pub(crate) fn mul(&mut self, other1: F64x4) {
        unsafe { self.mul_shadow(other1) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_shadow(&mut self, other1: F64x4) {
        self.val = _mm256_mul_pd(other1.val, self.val);
    }

    #[inline]
    pub(crate) fn mul_add_scalar(&mut self, other1: f64, other2: F64x4) {
        unsafe { self.mul_add_scalar_shadow(other1, other2) }
    }

    // self = other1 * other2 + self
    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn mul_add_scalar_shadow(&mut self, other1: f64, other2: F64x4) {
        let broadcast_other1: __m256d = _mm256_broadcast_sd(&other1);
        self.val = _mm256_fmadd_pd(broadcast_other1, other2.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar2(&mut self, other1: f64, other2: f64, other3: F64x4) {
        let mut v: MaybeUninit<__m256d> = MaybeUninit::<__m256d>::uninit();
        let b: *mut __m256d = v.as_mut_ptr();
        let b1: *mut f64 = b as *mut f64;
        let b2: *mut f64 = b1.add(1);
        let b3: *mut f64 = b1.add(2);
        let b4: *mut f64 = b1.add(3);
        *b1 = other1;
        *b2 = other1;
        *b3 = other2;
        *b4 = other2;
        self.val = _mm256_fmadd_pd(*b, other3.val, self.val);
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
}
