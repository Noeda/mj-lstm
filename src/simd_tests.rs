#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use crate::simd_aarch64::*;
    #[cfg(target_arch = "x86_64")]
    use crate::simd_amd64::*;
    use crate::simd_common::*;

    // mul_add_scalar

    quickcheck! {
        fn mul_add_scalar_works_F32x8(x1: f32, x2: f32, x3: f32) -> bool {
            unsafe {
                let expected_result = x1 + x2 * x3;

                let mut v = F32x8::new(x1, x1, x1, x1, x1, x1, x1, x1);
                let mut v3 = F32x8::new(x3, x3, x3, x3, x3, x3, x3, x3);
                v.mul_add_scalar(x2, v3);
                (v.v1() - expected_result).abs() < 0.001
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar_works_F64x4(x1: f64, x2: f64, x3: f64) -> bool {
            unsafe {
                let expected_result = x1 + x2 * x3;

                let mut v = F64x4::new(x1, x1, x1, x1);
                let mut v3 = F64x4::new(x3, x3, x3, x3);
                v.mul_add_scalar(x2, v3);
                (v.v1() - expected_result).abs() < 0.001
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar_works_F64x2(x1: f64, x2: f64, x3: f64) -> bool {
            unsafe {
                let expected_result1 = x1 + x2 * x3;
                let expected_result2 = x1 * 2.0 + x2 * x3 * 2.0;

                let mut v = F64x2::new(x1, x1*2.0);
                let mut v3 = F64x2::new(x3, x3*2.0);
                v.mul_add_scalar(x2, v3);
                (v.v1() - expected_result1).abs() < 0.001 &&
                  (v.v2() - expected_result2).abs() < 0.001
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar_works_F32x4(x1: f32, x2: f32, x3: f32) -> bool {
            unsafe {
                let expected_result = x1 + x2 * x3;

                let mut v = F32x4::new(x1, x1, x1, x1);
                let mut v3 = F32x4::new(x3, x3, x3, x3);
                v.mul_add_scalar(x2, v3);
                (v.v1() - expected_result).abs() < 0.001
            }
        }
    }

    // Fast sigmoid

    quickcheck! {
        fn fast_sigmoid_works_F32x8(x1: f32, x2: f32, x3: f32, x4: f32, x5: f32, x6: f32, x7:f32, x8:f32) -> bool {
            unsafe {
                let mut v = F32x8::new(x1, x2, x3, x4, x5, x6, x7, x8);
                v.fast_sigmoid();
                fast_sigmoid32(x1) == v.v1() &&
                fast_sigmoid32(x2) == v.v2() &&
                fast_sigmoid32(x3) == v.v3() &&
                fast_sigmoid32(x4) == v.v4() &&
                fast_sigmoid32(x5) == v.v5() &&
                fast_sigmoid32(x6) == v.v6() &&
                fast_sigmoid32(x7) == v.v7() &&
                fast_sigmoid32(x8) == v.v8()
            }
        }
    }

    quickcheck! {
        fn fast_sigmoid_works_F64x4(x1: f64, x2: f64, x3: f64, x4: f64) -> bool {
            unsafe {
                let mut v = F64x4::new(x1, x2, x3, x4);
                v.fast_sigmoid();
                fast_sigmoid(x1) == v.v1() &&
                fast_sigmoid(x2) == v.v2() &&
                fast_sigmoid(x3) == v.v3() &&
                fast_sigmoid(x4) == v.v4()
            }
        }
    }

    quickcheck! {
        fn fast_sigmoid_works_F64x2(x1: f64, x2: f64) -> bool {
            unsafe {
                let mut v = F64x2::new(x1, x2);
                v.fast_sigmoid();
                fast_sigmoid(x1) == v.v1() &&
                fast_sigmoid(x2) == v.v2()
            }
        }
    }

    quickcheck! {
        fn fast_sigmoid_works_F32x4(x1: f32, x2: f32, x3: f32, x4: f32) -> bool {
            unsafe {
                let mut v = F32x4::new(x1, x2, x3, x4);
                v.fast_sigmoid();
                fast_sigmoid32(x1) == v.v1() &&
                fast_sigmoid32(x2) == v.v2() &&
                fast_sigmoid32(x3) == v.v3() &&
                fast_sigmoid32(x4) == v.v4()
            }
        }
    }

    // mul_add_scalar2

    quickcheck! {
        fn mul_add_scalar2_works_F32x8(x1: f32, x2: f32, v1: f32, v2: f32) -> bool {
            unsafe {
                let mut vec1 = F32x8::new(v1, v1 * 2.0, v1 * 3.0, v1 * 4.0, v1 * 5.0, v1 * 6.0, v1 * 7.0, v1*8.0);
                let original = vec1.clone();
                let vec2 = F32x8::new(v2, v2 + 1.0, v2+2.0, v2+3.0, v2+4.0, v2+5.0, v2+6.0, v2+7.0);
                vec1.mul_add_scalar2(x1, x2, vec2);
                (vec1.v1() - (vec2.v1()*x1 + original.v1())).abs() < 0.001 &&
                (vec1.v2() - (vec2.v2()*x1 + original.v2())).abs() < 0.001 &&
                (vec1.v3() - (vec2.v3()*x1 + original.v3())).abs() < 0.001 &&
                (vec1.v4() - (vec2.v4()*x1 + original.v4())).abs() < 0.001 &&
                (vec1.v5() - (vec2.v5()*x2 + original.v5())).abs() < 0.001 &&
                (vec1.v6() - (vec2.v6()*x2 + original.v6())).abs() < 0.001 &&
                (vec1.v7() - (vec2.v7()*x2 + original.v7())).abs() < 0.001 &&
                (vec1.v8() - (vec2.v8()*x2 + original.v8())).abs() < 0.001
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar2_works_F64x4(x1: f64, x2: f64, v1: f64, v2: f64) -> bool {
            unsafe {
                let mut vec1 = F64x4::new(v1, v1 * 2.0, v1 * 3.0, v1 * 4.0);
                let original = vec1.clone();
                let vec2 = F64x4::new(v2, v2 + 1.0, v2+2.0, v2+3.0);
                vec1.mul_add_scalar2(x1, x2, vec2);
                (vec1.v1() - (vec2.v1()*x1 + original.v1())).abs() < 0.001 &&
                (vec1.v2() - (vec2.v2()*x1 + original.v2())).abs() < 0.001 &&
                (vec1.v3() - (vec2.v3()*x2 + original.v3())).abs() < 0.001 &&
                (vec1.v4() - (vec2.v4()*x2 + original.v4())).abs() < 0.001
            }
        }
    }

    quickcheck! {
        fn mul_add_scalar2_works_F32x4(x1: f32, x2: f32, v1: f32, v2: f32) -> bool {
            unsafe {
                let mut vec1 = F32x4::new(v1, v1 * 2.0, v1 * 3.0, v1 * 4.0);
                let original = vec1.clone();
                let vec2 = F32x4::new(v2, v2 + 1.0, v2+2.0, v2+3.0);
                vec1.mul_add_scalar2(x1, x2, vec2);
                (vec1.v1() - (vec2.v1()*x1 + original.v1())).abs() < 0.001 &&
                (vec1.v2() - (vec2.v2()*x1 + original.v2())).abs() < 0.001 &&
                (vec1.v3() - (vec2.v3()*x2 + original.v3())).abs() < 0.001 &&
                (vec1.v4() - (vec2.v4()*x2 + original.v4())).abs() < 0.001
            }
        }
    }

    // vX_vec functions
    quickcheck! {
        fn v1_vec_works_F64x4(v1: f64, v2: f64, v3: f64) -> bool {
            unsafe {
            let vec1 = F64x4::new(v3, v1, v1, v1);
            let vec2 = F64x4::new(v2, v2, v2, v2);
            let vec3 = F64x4::new(v1, v3, v3, v3);
            let result = F64x4::v1_vec(&[vec1, vec2, vec3]);
            result == &[v3, v2, v1]
            }
        }
    }

    quickcheck! {
        fn v2_vec_works_F64x4(v1: f64, v2: f64, v3: f64) -> bool {
            unsafe {
            let vec1 = F64x4::new(v1, v3, v1, v1);
            let vec2 = F64x4::new(v2, v2, v2, v2);
            let vec3 = F64x4::new(v3, v1, v3, v3);
            let result = F64x4::v2_vec(&[vec1, vec2, vec3]);
            result == &[v3, v2, v1]
            }
        }
    }

    quickcheck! {
        fn v3_vec_works_F64x4(v1: f64, v2: f64, v3: f64) -> bool {
            unsafe {
            let vec1 = F64x4::new(v1, v1, v3, v1);
            let vec2 = F64x4::new(v2, v2, v2, v2);
            let vec3 = F64x4::new(v3, v3, v1, v3);
            let result = F64x4::v3_vec(&[vec1, vec2, vec3]);
            result == &[v3, v2, v1]
            }
        }
    }

    quickcheck! {
        fn v4_vec_works_F64x4(v1: f64, v2: f64, v3: f64) -> bool {
            unsafe {
            let vec1 = F64x4::new(v1, v1, v1, v3);
            let vec2 = F64x4::new(v2, v2, v2, v2);
            let vec3 = F64x4::new(v3, v3, v3, v1);
            let result = F64x4::v4_vec(&[vec1, vec2, vec3]);
            result == &[v3, v2, v1]
            }
        }
    }

    quickcheck! {
        fn v2_vec_works_F64x2(v1: f64, v2: f64, v3: f64) -> bool {
            unsafe {
            let vec1 = F64x2::new(v1, v2);
            let vec2 = F64x2::new(v2, v3);
            let vec3 = F64x2::new(v3, v1);
            let result = F64x2::v2_vec(&[vec1, vec2, vec3]);
            result == &[v2, v3, v1]
            }
        }
    }

    quickcheck! {
        fn from_F64x2_works(v1: f64, v2: f64, v3: f64, v4: f64) -> bool {
            unsafe {
            let vec1 = F64x2::new(v1, v2);
            let vec2 = F64x2::new(v3, v4);
            let vec3 = F64x4::new(v1, v2, v3, v4);
            let vec4 = F64x4::from_F64x2(vec1, vec2);

            vec1.v1() == vec3.v1() && vec1.v2() == vec3.v2() &&
                vec2.v1() == vec3.v3() && vec2.v2() == vec3.v4() &&
                vec1.v1() == vec4.v1() && vec1.v2() == vec4.v2() &&
                vec2.v1() == vec4.v3() && vec2.v2() == vec4.v4()
            }
        }
    }
}
