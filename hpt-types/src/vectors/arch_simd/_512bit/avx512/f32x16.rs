use crate::arch_simd::sleef::arch::helper_avx512::vabs_vf_vf;
use crate::arch_simd::sleef::libm::sleefsimdsp::{
    xacosf_u1, xacoshf, xasinf_u1, xasinhf, xatan2f_u1, xatanf_u1, xatanhf, xcbrtf_u1, xcopysignf,
    xcosf_u1, xcoshf, xerff_u1, xexp10f, xexp2f, xexpf, xexpm1f, xhypotf_u05, xlog10f, xlog1pf,
    xlog2f, xlogf_u1, xmaxf, xminf, xpowf, xroundf, xsincosf_u1, xsinf_u1, xsinhf, xsqrtf_u05,
    xtanf_u1, xtanhf, xtruncf,
};
use crate::convertion::VecConvertor;
use crate::simd::sleef::libm::sleefsimdsp::{xceilf, xfloorf, xsigmoid};
use crate::traits::{SimdCompare, SimdMath, SimdSelect, VecTrait};
use crate::type_promote::Eval2;
use crate::vectors::arch_simd::_512bit::u32x16;

use std::arch::x86_64::*;

use crate::simd::_512bit::f32x16;
use crate::simd::_512bit::i32x16;
#[cfg(target_pointer_width = "32")]
use crate::simd::_512bit::isizex2;
#[cfg(target_pointer_width = "32")]
use crate::simd::_512bit::usizex2;

impl PartialEq for f32x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, other.0);
            cmp == 0xFFFF
        }
    }
}

impl Default for f32x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { f32x16(_mm512_setzero_ps()) }
    }
}

impl VecTrait<f32> for f32x16 {
    const SIZE: usize = 16;
    type Base = f32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(not(target_feature = "fma"))]
        unsafe {
            f32x16(_mm512_add_ps(_mm512_mul_ps(self.0, a.0), b.0))
        }
        #[cfg(target_feature = "fma")]
        unsafe {
            f32x16(_mm512_fmadd_ps(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        unsafe {
            _mm512_reduce_add_ps(self.0)
        }
    }
    #[inline(always)]
    fn splat(val: f32) -> f32x16 {
        unsafe { f32x16(_mm512_set1_ps(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const f32) -> Self {
        f32x16(_mm512_loadu_ps(ptr))
    }
}

impl SimdCompare for f32x16 {
    type SimdMask = i32x16;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, rhs.0);

            let all_ones = _mm512_set1_epi32(-1);
            let all_zeros = _mm512_setzero_si512();
            i32x16(_mm512_mask_mov_epi32(all_zeros, mask, all_ones))
        }
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_OQ>(self.0, rhs.0);
            let all_ones = _mm512_set1_epi32(-1);
            let all_zeros = _mm512_setzero_si512();
            i32x16(_mm512_mask_mov_epi32(all_zeros, mask, all_ones))
        }
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, rhs.0);
            let all_ones = _mm512_set1_epi32(-1);
            let all_zeros = _mm512_setzero_si512();
            i32x16(_mm512_mask_mov_epi32(all_zeros, mask, all_ones))
        }
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(self.0, rhs.0);
            let all_ones = _mm512_set1_epi32(-1);
            let all_zeros = _mm512_setzero_si512();
            i32x16(_mm512_mask_mov_epi32(all_zeros, mask, all_ones))
        }
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, rhs.0);
            let all_ones = _mm512_set1_epi32(-1);
            let all_zeros = _mm512_setzero_si512();
            i32x16(_mm512_mask_mov_epi32(all_zeros, mask, all_ones))
        }
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(self.0, rhs.0);
            let all_ones = _mm512_set1_epi32(-1);
            let all_zeros = _mm512_setzero_si512();
            i32x16(_mm512_mask_mov_epi32(all_zeros, mask, all_ones))
        }
    }
}

impl SimdSelect<f32x16> for i32x16 {
    #[inline(always)]
    fn select(&self, true_val: f32x16, false_val: f32x16) -> f32x16 {
        unsafe {
            let mask = _mm512_movepi32_mask(self.0);
            f32x16(_mm512_mask_blend_ps(mask, false_val.0, true_val.0))
        }
    }
}

impl std::ops::Add for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x16(_mm512_add_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Sub for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { f32x16(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x16(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Div for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { f32x16(_mm512_div_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Neg for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { f32x16(_mm512_xor_ps(self.0, _mm512_set1_ps(-0.0))) }
    }
}

impl SimdMath<f32> for f32x16 {
    #[inline(always)]
    fn sin(self) -> Self {
        f32x16(unsafe { xsinf_u1(self.0) })
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f32x16(unsafe { xcosf_u1(self.0) })
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f32x16(unsafe { xtanf_u1(self.0) })
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x16(unsafe { xsqrtf_u05(self.0) })
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32x16(unsafe { vabs_vf_vf(self.0) })
    }
    #[inline(always)]
    fn floor(self) -> Self {
        Self(unsafe { xfloorf(self.0) })
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        Self(unsafe { xceilf(self.0) })
    }
    #[inline(always)]
    fn neg(self) -> Self {
        f32x16(unsafe { _mm512_sub_ps(_mm512_setzero_ps(), self.0) })
    }
    #[inline(always)]
    fn round(self) -> Self {
        f32x16(unsafe { xroundf(self.0) })
    }
    #[inline(always)]
    fn signum(self) -> Self {
        unsafe {
            let zero = _mm512_set1_ps(0.0);
            let ones = _mm512_set1_ps(1.0);
            let neg_ones = _mm512_set1_ps(-1.0);

            let gt_mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, zero);
            let lt_mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, zero);

            let mut result = _mm512_setzero_ps();

            result = _mm512_mask_blend_ps(gt_mask, result, ones);

            result = _mm512_mask_blend_ps(lt_mask, result, neg_ones);

            f32x16(result)
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(f32x16::splat(0.0)) + alpha * self.min(f32x16::splat(0.0))
    }
    #[inline(always)]
    fn relu(self) -> Self {
        self.max(Self::splat(0.0))
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        self.max(Self::splat(0.0)).min(Self::splat(6.0))
    }
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        f32x16(unsafe { xpowf(self.0, exp.0) })
    }
    #[inline(always)]
    fn asin(self) -> Self {
        f32x16(unsafe { xasinf_u1(self.0) })
    }
    #[inline(always)]
    fn acos(self) -> Self {
        f32x16(unsafe { xacosf_u1(self.0) })
    }
    #[inline(always)]
    fn atan(self) -> Self {
        f32x16(unsafe { xatanf_u1(self.0) })
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        f32x16(unsafe { xsinhf(self.0) })
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        f32x16(unsafe { xcoshf(self.0) })
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        f32x16(unsafe { xtanhf(self.0) })
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        f32x16(unsafe { xasinhf(self.0) })
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        f32x16(unsafe { xacoshf(self.0) })
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        f32x16(unsafe { xatanhf(self.0) })
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        f32x16(unsafe { xexp2f(self.0) })
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        f32x16(unsafe { xexp10f(self.0) })
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        f32x16(unsafe { xexpm1f(self.0) })
    }
    #[inline(always)]
    fn log10(self) -> Self {
        f32x16(unsafe { xlog10f(self.0) })
    }
    #[inline(always)]
    fn log2(self) -> Self {
        f32x16(unsafe { xlog2f(self.0) })
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        f32x16(unsafe { xlog1pf(self.0) })
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        f32x16(unsafe { xhypotf_u05(self.0, other.0) })
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        f32x16(unsafe { xtruncf(self.0) })
    }
    #[inline(always)]
    fn erf(self) -> Self {
        f32x16(unsafe { xerff_u1(self.0) })
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        f32x16(unsafe { xcbrtf_u1(self.0) })
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f32x16(unsafe { xexpf(self.0) })
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f32x16(unsafe { xlogf_u1(self.0) })
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincosf_u1(self.0) };
        (f32x16(ret.x), f32x16(ret.y))
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f32x16(unsafe { xatan2f_u1(self.0, other.0) })
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32x16(unsafe { xminf(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32x16(unsafe { xmaxf(self.0, other.0) })
    }

    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let sixth = Self::splat(1.0 / 6.0);
        let half = Self::splat(0.5);
        let one = Self::splat(1.0);
        let zero = Self::splat(0.0);
        let result = self * sixth + half;
        result.min(one).max(zero)
    }

    #[inline(always)]
    fn elu(self, alpha: Self) -> Self {
        let mask = self.simd_gt(Self::splat(0.0));
        mask.select(self, alpha * (self.expm1()))
    }

    #[inline(always)]
    fn selu(self, alpha: Self, scale: Self) -> Self {
        scale * self.elu(alpha)
    }

    #[inline(always)]
    fn celu(self, scale: Self) -> Self {
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
        unsafe {
            // 计算正常情况下的倒数
            let one = _mm512_set1_ps(1.0);
            let recip = _mm512_div_ps(one, self.0);

            // 检查NaN
            let is_nan_mask = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);
            let nan = _mm512_set1_ps(f32::NAN);
            let result_after_nan = _mm512_mask_mov_ps(recip, is_nan_mask, nan);

            // 检查零
            let zero = _mm512_setzero_ps();
            let is_zero_mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, zero);
            let inf = _mm512_set1_ps(f32::INFINITY);
            let final_result = _mm512_mask_mov_ps(result_after_nan, is_zero_mask, inf);

            Self(final_result)
        }
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        unsafe { Self(xsigmoid(self.0)) }
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        self / (Self::splat(1.0) + self.abs())
    }
    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        unsafe { f32x16(xcopysignf(self.0, rhs.0)) }
    }
}

impl VecConvertor for f32x16 {
    #[inline(always)]
    fn to_u32(self) -> u32x16 {
        unsafe { u32x16(_mm512_cvtps_epi32(self.0)) }
    }
    #[inline(always)]
    fn to_i32(self) -> i32x16 {
        unsafe { i32x16(_mm512_cvtps_epi32(self.0)) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> isizex2 {
        self.to_i32().to_isize()
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> usizex2 {
        self.to_u32().to_usize()
    }
    #[inline(always)]
    fn to_f32(self) -> f32x16 {
        self
    }
}

impl Eval2 for f32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f32x16::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let i: i32x16 = unsafe { std::mem::transmute(self.0) };
        let sign_mask = i32x16::splat(-0x8000_0000i32);
        let inf_mask = i32x16::splat(0x7f80_0000i32);
        let frac_mask = i32x16::splat(0x007f_ffffi32);

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(i32x16::splat(0));
        let is_neg = (i & sign_mask).simd_ne(i32x16::splat(0));

        is_inf.select(
            is_neg.select(i32x16::splat(-1), i32x16::splat(1)),
            i32x16::splat(0),
        )
    }
}
