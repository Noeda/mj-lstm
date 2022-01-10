use crate::simd_common::*;
use core::arch::x86_64::*;
use std::mem::MaybeUninit;

#[derive(Copy, Clone)]
pub struct F64x2 {
    pub(crate) val: __m128d,
}

#[derive(Copy, Clone)]
pub struct F64x4 {
    pub(crate) val: __m256d,
}

#[derive(Copy, Clone)]
pub struct F32x8 {
    pub(crate) val: __m256,
}

#[derive(Copy, Clone)]
pub struct F32x4 {
    pub(crate) val: __m128,
}

impl F32x4 {
    pub(crate) unsafe fn new(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        F32x4 {
            val: _mm_set_ps(x1, x2, x3, x4),
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
        unsafe { std::mem::transmute::<__m128, [f32; 4]>(self.val)[3] }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f32 {
        unsafe { std::mem::transmute::<__m128, [f32; 4]>(self.val)[2] }
    }

    #[inline]
    pub(crate) fn v3(&self) -> f32 {
        unsafe { std::mem::transmute::<__m128, [f32; 4]>(self.val)[1] }
    }

    #[inline]
    pub(crate) fn v4(&self) -> f32 {
        unsafe { std::mem::transmute::<__m128, [f32; 4]>(self.val)[0] }
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
            val: _mm256_set_ps(x1, x2, x3, x4, x5, x6, x7, x8),
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
        *b2 = _mm_broadcast_ss(&other1);
        *b1 = _mm_broadcast_ss(&other2);
        self.val = _mm256_fmadd_ps(*b, other3.val, self.val);
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
    pub(crate) fn v1_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(7)
        }
    }

    #[inline]
    pub(crate) fn v2_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(6)
        }
    }

    #[inline]
    pub(crate) fn v3_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(5)
        }
    }

    #[inline]
    pub(crate) fn v4_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(4)
        }
    }

    #[inline]
    pub(crate) fn v5_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(3)
        }
    }

    #[inline]
    pub(crate) fn v6_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(2)
        }
    }

    #[inline]
    pub(crate) fn v7_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(1)
        }
    }

    #[inline]
    pub(crate) fn v8_mut(&mut self) -> &mut f32 {
        unsafe {
            std::mem::transmute::<&mut __m256, &mut [f32; 8]>(&mut self.val).get_unchecked_mut(0)
        }
    }

    #[inline]
    pub(crate) fn v1(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[7] }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[6] }
    }

    #[inline]
    pub(crate) fn v3(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[5] }
    }

    #[inline]
    pub(crate) fn v4(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[4] }
    }

    #[inline]
    pub(crate) fn v5(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[3] }
    }

    #[inline]
    pub(crate) fn v6(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[2] }
    }

    #[inline]
    pub(crate) fn v7(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[1] }
    }

    #[inline]
    pub(crate) fn v8(&self) -> f32 {
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(self.val)[0] }
    }
}

impl F64x4 {
    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn new(x1: f64, x2: f64, x3: f64, x4: f64) -> Self {
        F64x4 {
            val: _mm256_set_pd(x1, x2, x3, x4),
        }
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn from_F64x2(x1: F64x2, x2: F64x2) -> Self {
        F64x4 {
            val: _mm256_set_m128d(x1.val, x2.val),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn fast_sigmoid(&mut self) {
        let half = _mm256_broadcast_sd(&0.5);
        let one = _mm256_broadcast_sd(&1.0);
        let negzero = _mm256_broadcast_sd(&-0.0);
        let self_abs = _mm256_andnot_pd(negzero, self.val);
        let plus_one = _mm256_add_pd(one, self_abs);
        let xdivided = _mm256_div_pd(self.val, plus_one);
        self.val = _mm256_fmadd_pd(xdivided, half, half)
    }

    #[inline]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar(&mut self, other1: f64, other2: F64x4) {
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
        *b4 = other1;
        *b3 = other1;
        *b2 = other2;
        *b1 = other2;
        self.val = _mm256_fmadd_pd(*b, other3.val, self.val);
    }

    fn vec_from_vec64(x1: &[f64], x2: &[f64], x3: &[f64], x4: &[f64]) -> Vec<F64x4> {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x2.len(), x3.len());
        assert_eq!(x3.len(), x4.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(unsafe { F64x4::new(x1[idx], x2[idx], x3[idx], x4[idx]) });
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
    pub(crate) fn v1_mut(&mut self) -> &mut f64 {
        unsafe {
            std::mem::transmute::<&mut __m256d, &mut [f64; 4]>(&mut self.val).get_unchecked_mut(3)
        }
    }

    #[inline]
    pub(crate) fn v2_mut(&mut self) -> &mut f64 {
        unsafe {
            std::mem::transmute::<&mut __m256d, &mut [f64; 4]>(&mut self.val).get_unchecked_mut(2)
        }
    }

    #[inline]
    pub(crate) fn v3_mut(&mut self) -> &mut f64 {
        unsafe {
            std::mem::transmute::<&mut __m256d, &mut [f64; 4]>(&mut self.val).get_unchecked_mut(1)
        }
    }

    #[inline]
    pub(crate) fn v4_mut(&mut self) -> &mut f64 {
        unsafe {
            std::mem::transmute::<&mut __m256d, &mut [f64; 4]>(&mut self.val).get_unchecked_mut(0)
        }
    }

    #[inline]
    pub(crate) fn v1(&self) -> f64 {
        unsafe { std::mem::transmute::<__m256d, [f64; 4]>(self.val)[3] }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f64 {
        unsafe { std::mem::transmute::<__m256d, [f64; 4]>(self.val)[2] }
    }

    #[inline]
    pub(crate) fn v3(&self) -> f64 {
        unsafe { std::mem::transmute::<__m256d, [f64; 4]>(self.val)[1] }
    }

    #[inline]
    pub(crate) fn v4(&self) -> f64 {
        unsafe { std::mem::transmute::<__m256d, [f64; 4]>(self.val)[0] }
    }
}

impl F64x2 {
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn new(x1: f64, x2: f64) -> Self {
        F64x2 {
            val: _mm_set_pd(x1, x2),
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn fast_sigmoid(&mut self) {
        let half = _mm_set_pd1(0.5);
        let one = _mm_set_pd1(1.0);
        let negzero = _mm_set_pd1(-0.0);
        let self_abs = _mm_andnot_pd(negzero, self.val);
        let plus_one = _mm_add_pd(one, self_abs);
        let xdivided = _mm_div_pd(self.val, plus_one);
        self.val = _mm_fmadd_pd(xdivided, half, half)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar(&mut self, other1: f64, other2: F64x2) {
        let broadcast_other1: __m128d = _mm_set_pd1(other1);
        self.val = _mm_fmadd_pd(broadcast_other1, other2.val, self.val);
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    #[target_feature(enable = "fma")]
    pub(crate) unsafe fn mul_add_scalar_scalar(&mut self, other1: f64, other2: f64) {
        let broadcast_other1: __m128d = _mm_set_pd1(other1);
        let broadcast_other2: __m128d = _mm_set_pd1(other2);
        self.val = _mm_fmadd_pd(broadcast_other1, self.val, broadcast_other2);
    }

    fn vec_from_vec64(x1: &[f64], x2: &[f64]) -> Vec<F64x2> {
        assert_eq!(x1.len(), x2.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(unsafe { F64x2::new(x1[idx], x2[idx]) });
        }

        result
    }

    pub(crate) fn vecs_from_vecs64(x1: &[Vec<f64>], x2: &[Vec<f64>]) -> Vec<Vec<F64x2>> {
        assert_eq!(x1.len(), x2.len());

        let mut result = Vec::with_capacity(x1.len());
        for idx in 0..x1.len() {
            result.push(F64x2::vec_from_vec64(&x1[idx], &x2[idx]));
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

    #[inline]
    pub(crate) fn v1_mut(&mut self) -> &mut f64 {
        unsafe {
            std::mem::transmute::<&mut __m128d, &mut [f64; 2]>(&mut self.val).get_unchecked_mut(1)
        }
    }

    #[inline]
    pub(crate) fn v2_mut(&mut self) -> &mut f64 {
        unsafe {
            std::mem::transmute::<&mut __m128d, &mut [f64; 2]>(&mut self.val).get_unchecked_mut(0)
        }
    }

    #[inline]
    pub(crate) fn v1(&self) -> f64 {
        unsafe { std::mem::transmute::<__m128d, [f64; 2]>(self.val)[1] }
    }

    #[inline]
    pub(crate) fn v2(&self) -> f64 {
        unsafe { std::mem::transmute::<__m128d, [f64; 2]>(self.val)[0] }
    }
}
