use crate::arch_simd::sleef::arch::helper_avx2::vabs_vf_vf;
use crate::arch_simd::sleef::libm::sleefsimdsp::{
    xacosf_u1, xacoshf, xasinf_u1, xasinhf, xatan2f_u1, xatanf_u1, xatanhf, xcbrtf_u1, xcosf_u1,
    xcoshf, xerff_u1, xexp10f, xexp2f, xexpf, xexpm1f, xhypotf_u05, xlog10f, xlog1pf, xlog2f,
    xlogf_u1, xmaxf, xminf, xpowf, xroundf, xsincosf_u1, xsinf_u1, xsinhf, xsqrtf_u05, xtanf_u1,
    xtanhf, xtruncf,
};
use crate::convertion::VecConvertor;
use crate::traits::{SimdCompare, SimdMath, SimdSelect, VecTrait};
use crate::vectors::arch_simd::_256bit::u32x8::u32x8;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x8::i32x8;

/// a vector of 8 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f32x8(pub(crate) __m256);

impl PartialEq for f32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ);
            _mm256_movemask_ps(cmp) == 0xFF
        }
    }
}

impl Default for f32x8 {
    fn default() -> Self {
        unsafe { f32x8(_mm256_setzero_ps()) }
    }
}

impl VecTrait<f32> for f32x8 {
    const SIZE: usize = 8;
    type Base = f32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        unsafe {
            _mm256_storeu_ps(
                &mut self.0 as *mut _ as *mut f32,
                _mm256_loadu_ps(slice.as_ptr()),
            );
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
        unsafe {
            f32x8(_mm256_add_ps(_mm256_mul_ps(self.0, a.0), b.0))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        unsafe {
            f32x8(_mm256_fmadd_ps(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        unsafe {
            let sum = _mm256_hadd_ps(self.0, self.0);
            let sum = _mm256_hadd_ps(sum, sum);
            _mm_cvtss_f32(_mm256_castps256_ps128(sum))
        }
    }
    fn splat(val: f32) -> f32x8 {
        unsafe { f32x8(_mm256_set1_ps(val)) }
    }
}

impl f32x8 {
    #[allow(unused)]
    fn as_array(&self) -> [f32; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// check if the vector is nan
    pub fn is_nan(&self) -> f32x8 {
        unsafe { f32x8(_mm256_cmp_ps(self.0, self.0, _CMP_UNORD_Q)) }
    }
    /// check if the vector is infinite
    pub fn is_infinite(&self) -> f32x8 {
        unsafe {
            let abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0);
            f32x8(_mm256_cmp_ps(
                abs,
                _mm256_set1_ps(f32::INFINITY),
                _CMP_EQ_OQ,
            ))
        }
    }
    /// reciprocal of the vector
    pub fn recip(&self) -> f32x8 {
        unsafe { f32x8(_mm256_rcp_ps(self.0)) }
    }
}

impl SimdCompare for f32x8 {
    type SimdMask = i32x8;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_EQ_OQ,
            )))
        }
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0,
                rhs.0,
                _CMP_NEQ_OQ,
            )))
        }
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_LT_OQ,
            )))
        }
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_LE_OQ,
            )))
        }
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_GT_OQ,
            )))
        }
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_GE_OQ,
            )))
        }
    }
}

impl SimdSelect<f32x8> for i32x8 {
    fn select(&self, true_val: f32x8, false_val: f32x8) -> f32x8 {
        unsafe {
            f32x8(_mm256_blendv_ps(
                false_val.0,
                true_val.0,
                std::mem::transmute(self.0),
            ))
        }
    }
}

impl std::ops::Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_div_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Rem for f32x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [f32; 8] = std::mem::transmute(self.0);
            let b: [f32; 8] = std::mem::transmute(rhs.0);
            let c: [f32; 8] = [
                a[0] % b[0],
                a[1] % b[1],
                a[2] % b[2],
                a[3] % b[3],
                a[4] % b[4],
                a[5] % b[5],
                a[6] % b[6],
                a[7] % b[7],
            ];
            f32x8(std::mem::transmute(c))
        }
    }
}
impl std::ops::Neg for f32x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe { f32x8(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
    }
}

impl SimdMath<f32> for f32x8 {
    fn sin(self) -> Self {
        f32x8(unsafe { xsinf_u1(self.0) })
    }
    fn cos(self) -> Self {
        f32x8(unsafe { xcosf_u1(self.0) })
    }
    fn tan(self) -> Self {
        f32x8(unsafe { xtanf_u1(self.0) })
    }

    fn square(self) -> Self {
        f32x8(unsafe { _mm256_mul_ps(self.0, self.0) })
    }

    fn sqrt(self) -> Self {
        f32x8(unsafe { xsqrtf_u05(self.0) })
    }

    fn abs(self) -> Self {
        f32x8(unsafe { vabs_vf_vf(self.0) })
    }

    fn floor(self) -> Self {
        f32x8(unsafe { _mm256_floor_ps(self.0) })
    }

    fn ceil(self) -> Self {
        f32x8(unsafe { _mm256_ceil_ps(self.0) })
    }

    fn neg(self) -> Self {
        f32x8(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) })
    }

    fn round(self) -> Self {
        f32x8(unsafe { xroundf(self.0) })
    }

    fn sign(self) -> Self {
        f32x8(unsafe { _mm256_and_ps(self.0, _mm256_set1_ps(0.0f32)) })
    }

    fn leaky_relu(self, _: f32) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        f32x8(unsafe { _mm256_max_ps(self.0, _mm256_setzero_ps()) })
    }

    fn relu6(self) -> Self {
        f32x8(unsafe { _mm256_min_ps(self.relu().0, _mm256_set1_ps(6.0f32)) })
    }

    fn pow(self, exp: Self) -> Self {
        f32x8(unsafe { xpowf(self.0, exp.0) })
    }

    fn asin(self) -> Self {
        f32x8(unsafe { xasinf_u1(self.0) })
    }

    fn acos(self) -> Self {
        f32x8(unsafe { xacosf_u1(self.0) })
    }

    fn atan(self) -> Self {
        f32x8(unsafe { xatanf_u1(self.0) })
    }

    fn sinh(self) -> Self {
        f32x8(unsafe { xsinhf(self.0) })
    }

    fn cosh(self) -> Self {
        f32x8(unsafe { xcoshf(self.0) })
    }

    fn tanh(self) -> Self {
        f32x8(unsafe { xtanhf(self.0) })
    }

    fn asinh(self) -> Self {
        f32x8(unsafe { xasinhf(self.0) })
    }

    fn acosh(self) -> Self {
        f32x8(unsafe { xacoshf(self.0) })
    }

    fn atanh(self) -> Self {
        f32x8(unsafe { xatanhf(self.0) })
    }

    fn exp2(self) -> Self {
        f32x8(unsafe { xexp2f(self.0) })
    }

    fn exp10(self) -> Self {
        f32x8(unsafe { xexp10f(self.0) })
    }

    fn expm1(self) -> Self {
        f32x8(unsafe { xexpm1f(self.0) })
    }

    fn log10(self) -> Self {
        f32x8(unsafe { xlog10f(self.0) })
    }

    fn log2(self) -> Self {
        f32x8(unsafe { xlog2f(self.0) })
    }

    fn log1p(self) -> Self {
        f32x8(unsafe { xlog1pf(self.0) })
    }

    fn hypot(self, other: Self) -> Self {
        f32x8(unsafe { xhypotf_u05(self.0, other.0) })
    }

    fn trunc(self) -> Self {
        f32x8(unsafe { xtruncf(self.0) })
    }

    fn erf(self) -> Self {
        f32x8(unsafe { xerff_u1(self.0) })
    }

    fn cbrt(self) -> Self {
        f32x8(unsafe { xcbrtf_u1(self.0) })
    }

    fn exp(self) -> Self {
        f32x8(unsafe { xexpf(self.0) })
    }

    fn ln(self) -> Self {
        f32x8(unsafe { xlogf_u1(self.0) })
    }

    fn log(self) -> Self {
        f32x8(unsafe { xlogf_u1(self.0) })
    }

    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincosf_u1(self.0) };
        (f32x8(ret.x), f32x8(ret.y))
    }

    fn atan2(self, other: Self) -> Self {
        f32x8(unsafe { xatan2f_u1(self.0, other.0) })
    }

    fn min(self, other: Self) -> Self {
        f32x8(unsafe { xminf(self.0, other.0) })
    }

    fn max(self, other: Self) -> Self {
        f32x8(unsafe { xmaxf(self.0, other.0) })
    }
}

impl VecConvertor for f32x8 {
    fn to_u32(self) -> super::u32x8::u32x8 {
        unsafe { u32x8(_mm256_cvtps_epi32(self.0)) }
    }
    fn to_i32(self) -> super::i32x8::i32x8 {
        unsafe { i32x8(_mm256_cvtps_epi32(self.0)) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> super::isizex2::isizex2 {
        self.to_i32().to_isize()
    }
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> super::usizex2::usizex2 {
        self.to_u32().to_usize()
    }
    fn to_f32(self) -> f32x8 {
        self
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::arch_simd::sleef::common::misc::SQRT_FLT_MAX;

//     use super::*;
//     use rug::Assign;
//     use rug::Float;
//     const TEST_REPEAT: usize = 100_000;
//     const PRECF32: u32 = 80;
//     pub fn f32_count_ulp(d: f32, c: &Float) -> f32 {
//         let c2 = c.to_f32();

//         if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
//             return 0.;
//         }

//         if (c2 == 0.) && (d != 0.) {
//             return 10000.;
//         }

//         if c2.is_infinite() && d.is_infinite() {
//             return 0.;
//         }

//         let prec = c.prec();

//         let mut fry = Float::with_val(prec, d);

//         let mut frw = Float::new(prec);

//         let (_, e) = c.to_f32_exp();

//         frw.assign(Float::u_exp(1, e - 24_i32));

//         fry -= c;
//         fry /= &frw;
//         let u = f32::from_bits(0x_7fff_ffff & fry.to_f32().to_bits());

//         u
//     }
//     fn f32_gen_input(
//         rng: &mut rand::rngs::ThreadRng,
//         range: core::ops::RangeInclusive<f32>,
//     ) -> f32 {
//         use rand::Rng;
//         let mut start = *range.start();
//         if start == f32::MIN {
//             start = -1e37;
//         }
//         let mut end = *range.end();
//         if end == f32::MAX {
//             end = 1e37;
//         }
//         rng.gen_range(start..=end)
//     }
//     fn gen_input(
//         rng: &mut rand::rngs::ThreadRng,
//         range: core::ops::RangeInclusive<f32>,
//     ) -> __m256
//     {
//         let mut arr = [0.; 8];
//         for i in 0..8 {
//             arr[i] = f32_gen_input(rng, range.clone());
//         }
//         unsafe { std::mem::transmute(arr) }
//     }
//     pub fn test_f_f(
//         f_tested: fn(__m256) -> __m256,
//         f_sample: fn(rug::Float) -> rug::Float,
//         range: core::ops::RangeInclusive<f32>,
//         ulp_ex: f32,
//         name: &str,
//     )
//     {
//         test_c_f_f(
//             f_tested,
//             f_sample,
//             range,
//             |ulp, _, _| (ulp <= ulp_ex, format!("ULP: {ulp} > {ulp_ex}")),
//             name,
//         )
//     }

//     pub fn test_c_f_f(
//         f_tested: fn(__m256) -> __m256,
//         f_sample: fn(rug::Float) -> rug::Float,
//         range: core::ops::RangeInclusive<f32>,
//         cf: impl Fn(f32, f32, &rug::Float) -> (bool, String),
//         name: &str,
//     )
//     {
//         let mut rng = rand::thread_rng();
//         for n in 0..TEST_REPEAT {
//             let input = gen_input(&mut rng, range.clone());
//             let in_fx: [f32; 8] = unsafe { std::mem::transmute(input) };
//             let out_fx: [f32; 8] = unsafe { std::mem::transmute(f_tested(input)) };
//             for i in 0..8 {
//                 let input = in_fx[i];
//                 let output = out_fx[i];
//                 let expected = f_sample(rug::Float::with_val(PRECF32, input));
//                 if expected.is_nan() && output.is_nan() {
//                     continue;
//                 }
//                 let ulp = f32_count_ulp(output, &expected);
//                 let (b, fault_string) = cf(ulp, output, &expected);
//                 assert!(
//                     b,
//                     "{}: Iteration: {n}, Position: {i}, Input: {input:e}, Output: {output}, Expected: {expected}, {}",
//                     name,
//                     fault_string
//                 );
//             }
//         }
//     }
//     #[test]
//     fn tests() {
//         macro_rules! define_func {
//             ($func:ident, $f:ident, $x_func:expr, $range:expr) => {
//                 fn $func(d: __m256) -> __m256
//                 {
//                     unsafe { $x_func(d).into() }
//                 }
//                 test_f_f($func, rug::Float::$f, $range, 1., stringify!($func));
//             };
//         }
//         define_func!(sinf, sin, xsinf_u1, f32::MIN..=f32::MAX);
//         define_func!(cosf, cos, xcosf_u1, f32::MIN..=f32::MAX);
//         define_func!(tanf, tan, xtanf_u1, f32::MIN..=f32::MAX);
//         define_func!(asin, asin, xasinf_u1, f32::MIN..=f32::MAX);
//         define_func!(acos, acos, xacosf_u1, f32::MIN..=f32::MAX);
//         define_func!(atan, atan, xatanf_u1, f32::MIN..=f32::MAX);
//         define_func!(sinh, sinh, xsinhf, -88.5..=88.5);
//         define_func!(cosh, cosh, xcoshf, -88.5..=88.5);
//         define_func!(tanh, tanh, xtanhf, -8.7..=8.7);
//         define_func!(
//             asinh,
//             asinh,
//             xasinhf,
//             -SQRT_FLT_MAX as f32..=SQRT_FLT_MAX as f32
//         );
//         define_func!(
//             acosh,
//             acosh,
//             xacoshf,
//             -SQRT_FLT_MAX as f32..=SQRT_FLT_MAX as f32
//         );
//         define_func!(atanh, atanh, xatanhf, f32::MIN..=f32::MAX);
//         define_func!(round, round, xroundf, f32::MIN..=f32::MAX);
//         define_func!(sqrt, sqrt, xsqrtf_u05, f32::MIN..=f32::MAX);
//         define_func!(exp, exp, xexpf, -104.0..=100.0);
//         define_func!(exp2, exp2, xexp2f, -150.0..=128.0);
//         define_func!(exp10, exp10, xexp10f, -50.0..=38.54);
//         define_func!(expm1, exp_m1, xexpm1f, -16.64..=88.73);
//         define_func!(log10, log10, xlog10f, 0.0..=f32::MAX);
//         define_func!(log2, log2, xlog2f, 0.0..=f32::MAX);
//         define_func!(log1p, ln_1p, xlog1pf, -1.0..=1e+38);
//         define_func!(trunc, trunc, xtruncf, f32::MIN..=f32::MAX);
//         define_func!(erf, erf, xerff_u1, f32::MIN..=f32::MAX);
//         define_func!(cbrt, cbrt, xcbrtf_u1, f32::MIN..=f32::MAX);
//         define_func!(ln, ln, xlogf_u1, 0.0..=f32::MAX);
//     }
// }
