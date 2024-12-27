use crate::arch_simd::sleef::arch::helper_avx2::vabs_vf_vf;
use crate::arch_simd::sleef::libm::sleefsimdsp::{
    xacosf_u1, xacoshf, xasinf_u1, xasinhf, xatan2f_u1, xatanf_u1, xatanhf, xcbrtf_u1, xcosf_u1,
    xcoshf, xerff_u1, xexp10f, xexp2f, xexpf, xexpm1f, xhypotf_u05, xlog10f, xlog1pf, xlog2f,
    xlogf_u1, xmaxf, xminf, xpowf, xroundf, xsincosf_u1, xsinf_u1, xsinhf, xsqrtf_u05, xtanf_u1,
    xtanhf, xtruncf,
};
use crate::convertion::VecConvertor;
use crate::traits::{SimdCompare, SimdMath, SimdSelect, VecTrait};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_256bit::u32x8::u32x8;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x8::i32x8;

/// a vector of 8 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f32x8(pub(crate) __m256);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type f32_promote = f32x8;

impl PartialEq for f32x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ);
            _mm256_movemask_ps(cmp) == 0xFF
        }
    }
}

impl Default for f32x8 {
    #[inline(always)]
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
    #[inline(always)]
    fn splat(val: f32) -> f32x8 {
        unsafe { f32x8(_mm256_set1_ps(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const f32) -> Self {
        f32x8(_mm256_loadu_ps(ptr))
    }
}

impl f32x8 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [f32; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// check if the vector is nan
    #[inline(always)]
    pub fn is_nan(&self) -> f32x8 {
        unsafe { f32x8(_mm256_cmp_ps(self.0, self.0, _CMP_UNORD_Q)) }
    }
    /// check if the vector is infinite
    #[inline(always)]
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
    #[inline(always)]
    pub fn recip(&self) -> f32x8 {
        unsafe { f32x8(_mm256_rcp_ps(self.0)) }
    }
}

impl SimdCompare for f32x8 {
    type SimdMask = i32x8;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_EQ_OQ,
            )))
        }
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0,
                rhs.0,
                _CMP_NEQ_OQ,
            )))
        }
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_LT_OQ,
            )))
        }
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_LE_OQ,
            )))
        }
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_GT_OQ,
            )))
        }
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            i32x8(_mm256_castps_si256(_mm256_cmp_ps(
                self.0, rhs.0, _CMP_GE_OQ,
            )))
        }
    }
}

impl SimdSelect<f32x8> for i32x8 {
    #[inline(always)]
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
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Sub for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Div for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_div_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Rem for f32x8 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { f32x8(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
    }
}

impl SimdMath<f32> for f32x8 {
    #[inline(always)]
    fn sin(self) -> Self {
        f32x8(unsafe { xsinf_u1(self.0) })
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f32x8(unsafe { xcosf_u1(self.0) })
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f32x8(unsafe { xtanf_u1(self.0) })
    }
    #[inline(always)]
    fn square(self) -> Self {
        f32x8(unsafe { _mm256_mul_ps(self.0, self.0) })
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x8(unsafe { xsqrtf_u05(self.0) })
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32x8(unsafe { vabs_vf_vf(self.0) })
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f32x8(unsafe { _mm256_floor_ps(self.0) })
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f32x8(unsafe { _mm256_ceil_ps(self.0) })
    }
    #[inline(always)]
    fn neg(self) -> Self {
        f32x8(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) })
    }
    #[inline(always)]
    fn round(self) -> Self {
        f32x8(unsafe { xroundf(self.0) })
    }
    #[inline(always)]
    fn sign(self) -> Self {
        f32x8(unsafe { _mm256_and_ps(self.0, _mm256_set1_ps(0.0f32)) })
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: f32) -> Self {
        let alpha = f32x8::splat(alpha);
        self.max(f32x8::splat(0.0)) + alpha * self.min(f32x8::splat(0.0))
    }
    #[inline(always)]
    fn relu(self) -> Self {
        f32x8(unsafe { _mm256_max_ps(self.0, _mm256_setzero_ps()) })
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        f32x8(unsafe { _mm256_min_ps(self.relu().0, _mm256_set1_ps(6.0f32)) })
    }
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        f32x8(unsafe { xpowf(self.0, exp.0) })
    }
    #[inline(always)]
    fn asin(self) -> Self {
        f32x8(unsafe { xasinf_u1(self.0) })
    }
    #[inline(always)]
    fn acos(self) -> Self {
        f32x8(unsafe { xacosf_u1(self.0) })
    }
    #[inline(always)]
    fn atan(self) -> Self {
        f32x8(unsafe { xatanf_u1(self.0) })
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        f32x8(unsafe { xsinhf(self.0) })
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        f32x8(unsafe { xcoshf(self.0) })
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        f32x8(unsafe { xtanhf(self.0) })
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        f32x8(unsafe { xasinhf(self.0) })
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        f32x8(unsafe { xacoshf(self.0) })
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        f32x8(unsafe { xatanhf(self.0) })
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        f32x8(unsafe { xexp2f(self.0) })
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        f32x8(unsafe { xexp10f(self.0) })
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        f32x8(unsafe { xexpm1f(self.0) })
    }
    #[inline(always)]
    fn log10(self) -> Self {
        f32x8(unsafe { xlog10f(self.0) })
    }
    #[inline(always)]
    fn log2(self) -> Self {
        f32x8(unsafe { xlog2f(self.0) })
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        f32x8(unsafe { xlog1pf(self.0) })
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        f32x8(unsafe { xhypotf_u05(self.0, other.0) })
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        f32x8(unsafe { xtruncf(self.0) })
    }
    #[inline(always)]
    fn erf(self) -> Self {
        f32x8(unsafe { xerff_u1(self.0) })
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        f32x8(unsafe { xcbrtf_u1(self.0) })
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f32x8(unsafe { xexpf(self.0) })
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f32x8(unsafe { xlogf_u1(self.0) })
    }
    #[inline(always)]
    fn log(self) -> Self {
        f32x8(unsafe { xlogf_u1(self.0) })
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincosf_u1(self.0) };
        (f32x8(ret.x), f32x8(ret.y))
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f32x8(unsafe { xatan2f_u1(self.0, other.0) })
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32x8(unsafe { xminf(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32x8(unsafe { xmaxf(self.0, other.0) })
    }
    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let point_two = Self::splat(0.2);
        let half = Self::splat(0.5);
        let one = Self::splat(1.0);
        let zero = Self::splat(0.0);
        let add = point_two * self + half;
        add.min(one).max(zero)
    }

    #[inline(always)]
    fn fast_hard_sigmoid(self) -> Self {
        let sixth = Self::splat(1.0 / 6.0);
        let half = Self::splat(0.5);
        let one = Self::splat(1.0);
        let zero = Self::splat(0.0);
        let result = self * sixth + half;
        result.min(one).max(zero)
    }

    #[inline(always)]
    fn elu(self, alpha: f32) -> Self {
        let mask = self.simd_gt(Self::splat(0.0));
        mask.select(self, Self::splat(alpha) * (self.expm1()))
    }

    #[inline(always)]
    fn selu(self, alpha: f32, scale: f32) -> Self {
        let scale = Self::splat(scale);
        scale * self.elu(alpha)
    }

    #[inline(always)]
    fn celu(self, alpha: f32) -> Self {
        let scale = Self::splat(alpha);
        let gt_mask = self.simd_gt(Self::splat(0.0));
        gt_mask.select(self, scale * (self.exp() - Self::splat(1.0)))
    }

    #[inline(always)]
    fn gelu(self) -> Self {
        let erf = (self * Self::splat(std::f32::consts::FRAC_1_SQRT_2)).erf() + Self::splat(1.0);
        let half = Self::splat(0.5);
        half * self * erf
    }

    #[inline(always)]
    fn hard_swish(self) -> Self {
        let three = Self::splat(3.0);
        self * (self + three).relu6() * Self::splat(1.0 / 6.0)
    }

    #[inline(always)]
    fn mish(self) -> Self {
        self * self.softplus().tanh()
    }

    #[inline(always)]
    fn softplus(self) -> Self {
        let one = Self::splat(1.0);
        (one + self.exp()).ln()
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self(unsafe {
            let is_nan = _mm256_cmp_ps(self.0, self.0, _CMP_UNORD_Q);
            let is_zero = _mm256_cmp_ps(self.0, _mm256_setzero_ps(), _CMP_EQ_OQ);
            let recip = _mm256_div_ps(_mm256_set1_ps(1.0), self.0);
            _mm256_blendv_ps(
                recip,
                _mm256_or_ps(
                    _mm256_and_ps(is_zero, _mm256_set1_ps(f32::INFINITY)),
                    _mm256_and_ps(is_nan, _mm256_set1_ps(f32::NAN)),
                ),
                _mm256_or_ps(is_nan, is_zero),
            )
        })
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        Self::splat(1.0) / (Self::splat(1.0) + (-self).exp())
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        self / (Self::splat(1.0) + self.abs())
    }
}

impl VecConvertor for f32x8 {
    #[inline(always)]
    fn to_u32(self) -> super::u32x8::u32x8 {
        unsafe { u32x8(_mm256_cvtps_epi32(self.0)) }
    }
    #[inline(always)]
    fn to_i32(self) -> super::i32x8::i32x8 {
        unsafe { i32x8(_mm256_cvtps_epi32(self.0)) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> super::isizex2::isizex2 {
        self.to_i32().to_isize()
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> super::usizex2::usizex2 {
        self.to_u32().to_usize()
    }
    #[inline(always)]
    fn to_f32(self) -> f32x8 {
        self
    }
}

impl FloatOutBinary2 for f32x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [
            self[0].log(base[0]),
            self[1].log(base[1]),
            self[2].log(base[2]),
            self[3].log(base[3]),
            self[4].log(base[4]),
            self[5].log(base[5]),
            self[6].log(base[6]),
            self[7].log(base[7]),
        ];
        f32x8(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for f32x8 {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        self.pow(rhs)
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clip(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for f32x8 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self.ceil()
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn __sign(self) -> Self {
        self.sign()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(f32x8::splat(0.0)) + alpha * self.min(f32x8::splat(0.0))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }
}

impl Eval2 for f32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe { std::mem::transmute(self.is_nan()) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unreachable!()
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        unsafe { std::mem::transmute(self.is_infinite()) }
    }
}
