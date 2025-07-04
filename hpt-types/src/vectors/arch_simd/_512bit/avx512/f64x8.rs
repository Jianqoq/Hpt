use crate::{
    arch_simd::sleef::{
        arch::helper_avx512::vabs_vd_vd,
        libm::sleefsimddp::{
            xacos_u1, xacosh, xasin_u1, xasinh, xatan2_u1, xatan_u1, xatanh, xcbrt_u1, xcopysign,
            xcos_u1, xcosh, xerf_u1, xexp, xexp10, xexp2, xexpm1, xfmax, xfmin, xhypot_u05, xlog10,
            xlog1p, xlog2, xlog_u1, xpow, xround, xsin_u1, xsincos_u1, xsinh, xsqrt_u05, xtan_u1,
            xtanh, xtrunc,
        },
    },
    convertion::VecConvertor,
    simd::sleef::libm::sleefsimddp::{xceil, xfloor},
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::Eval2,
};

use std::arch::x86_64::*;

use crate::simd::_512bit::f64x8;
use crate::simd::_512bit::i64x8;
#[cfg(target_pointer_width = "64")]
use crate::simd::_512bit::isizex8;
use crate::simd::_512bit::u64x8;
#[cfg(target_pointer_width = "64")]
use crate::simd::_512bit::usizex8;

impl PartialEq for f64x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, other.0);
            mask == 0xff
        }
    }
}

impl Default for f64x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { f64x8(_mm512_setzero_pd()) }
    }
}

impl VecTrait<f64> for f64x8 {
    const SIZE: usize = 8;
    type Base = f64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(not(target_feature = "fma"))]
        unsafe {
            f64x8(_mm512_add_pd(_mm512_mul_pd(self.0, a.0), b.0))
        }
        #[cfg(target_feature = "fma")]
        unsafe {
            f64x8(_mm512_fmadd_pd(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        unsafe { _mm512_reduce_add_pd(self.0) }
    }
    #[inline(always)]
    fn splat(val: f64) -> f64x8 {
        unsafe { f64x8(_mm512_set1_pd(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const f64) -> Self {
        f64x8(_mm512_loadu_pd(ptr))
    }
}

impl SimdCompare for f64x8 {
    type SimdMask = i64x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_OQ>(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
}

impl SimdSelect<f64x8> for i64x8 {
    #[inline(always)]
    fn select(&self, true_val: f64x8, false_val: f64x8) -> f64x8 {
        unsafe {
            let mask = _mm512_movepi64_mask(self.0);
            f64x8(_mm512_mask_blend_pd(mask, false_val.0, true_val.0))
        }
    }
}

impl std::ops::Add for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f64x8(_mm512_add_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { f64x8(_mm512_sub_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f64x8(_mm512_mul_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Div for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { f64x8(_mm512_div_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Rem for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let x: [f64; 8] = std::mem::transmute(self.0);
            let y: [f64; 8] = std::mem::transmute(rhs.0);
            let result = [
                x[0] % y[0],
                x[1] % y[1],
                x[2] % y[2],
                x[3] % y[3],
                x[4] % y[4],
                x[5] % y[5],
                x[6] % y[6],
                x[7] % y[7],
            ];
            f64x8(_mm512_loadu_pd(result.as_ptr()))
        }
    }
}
impl std::ops::Neg for f64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f64x8(_mm512_xor_pd(self.0, _mm512_set1_pd(-0.0))) }
    }
}

impl SimdMath<f64> for f64x8 {
    #[inline(always)]
    fn sin(self) -> Self {
        f64x8(unsafe { xsin_u1(self.0) })
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f64x8(unsafe { xcos_u1(self.0) })
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f64x8(unsafe { xtan_u1(self.0) })
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64x8(unsafe { xsqrt_u05(self.0) })
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64x8(unsafe { vabs_vd_vd(self.0) })
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f64x8(unsafe { xfloor(self.0) })
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f64x8(unsafe { xceil(self.0) })
    }
    #[inline(always)]
    fn neg(self) -> Self {
        f64x8(unsafe { _mm512_sub_pd(_mm512_setzero_pd(), self.0) })
    }
    #[inline(always)]
    fn round(self) -> Self {
        f64x8(unsafe { xround(self.0) })
    }
    #[inline(always)]
    fn signum(self) -> Self {
        unsafe {
            let zero = Self::splat(0.0);
            let ones = Self::splat(1.0);
            let neg_ones = Self::splat(-1.0);

            let gt_mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(self.0, zero.0);
            let lt_mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(self.0, zero.0);

            let pos = _mm512_maskz_mov_pd(gt_mask, ones.0);
            let neg = _mm512_maskz_mov_pd(lt_mask, neg_ones.0);

            f64x8(_mm512_add_pd(pos, neg))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0.0)) + alpha * self.min(Self::splat(0.0))
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
        f64x8(unsafe { xpow(self.0, exp.0) })
    }
    #[inline(always)]
    fn asin(self) -> Self {
        f64x8(unsafe { xasin_u1(self.0) })
    }
    #[inline(always)]
    fn acos(self) -> Self {
        f64x8(unsafe { xacos_u1(self.0) })
    }
    #[inline(always)]
    fn atan(self) -> Self {
        f64x8(unsafe { xatan_u1(self.0) })
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        f64x8(unsafe { xsinh(self.0) })
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        f64x8(unsafe { xcosh(self.0) })
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        f64x8(unsafe { xtanh(self.0) })
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        f64x8(unsafe { xasinh(self.0) })
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        f64x8(unsafe { xacosh(self.0) })
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        f64x8(unsafe { xatanh(self.0) })
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        f64x8(unsafe { xexp2(self.0) })
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        f64x8(unsafe { xexp10(self.0) })
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        f64x8(unsafe { xexpm1(self.0) })
    }
    #[inline(always)]
    fn log10(self) -> Self {
        f64x8(unsafe { xlog10(self.0) })
    }
    #[inline(always)]
    fn log2(self) -> Self {
        f64x8(unsafe { xlog2(self.0) })
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        f64x8(unsafe { xlog1p(self.0) })
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        f64x8(unsafe { xhypot_u05(self.0, other.0) })
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        f64x8(unsafe { xtrunc(self.0) })
    }
    #[inline(always)]
    fn erf(self) -> Self {
        f64x8(unsafe { xerf_u1(self.0) })
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        f64x8(unsafe { xcbrt_u1(self.0) })
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64x8(unsafe { xexp(self.0) })
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f64x8(unsafe { xlog_u1(self.0) })
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f64x8(unsafe { xatan2_u1(self.0, other.0) })
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincos_u1(self.0) };
        (f64x8(ret.x), f64x8(ret.y))
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64x8(unsafe { xfmin(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64x8(unsafe { xfmax(self.0, other.0) })
    }

    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let sixth = f64x8::splat(1.0 / 6.0);
        let half = f64x8::splat(0.5);
        let one = f64x8::splat(1.0);
        let zero = f64x8::splat(0.0);
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
        let erf = (self * Self::splat(std::f64::consts::FRAC_1_SQRT_2)).erf() + Self::splat(1.0);
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
            let is_nan_mask = _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(self.0, self.0);
            let is_zero_mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, _mm512_setzero_pd());

            let recip = _mm512_div_pd(_mm512_set1_pd(1.0), self.0);

            let inf = _mm512_set1_pd(f64::INFINITY);
            let nan = _mm512_set1_pd(f64::NAN);

            let special_cases =
                _mm512_mask_blend_pd(is_zero_mask, _mm512_maskz_mov_pd(is_nan_mask, nan), inf);

            Self(_mm512_mask_blend_pd(
                is_nan_mask | is_zero_mask,
                recip,
                special_cases,
            ))
        }
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        Self::splat(1.0) / (Self::splat(1.0) + (-self).exp())
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        self / (Self::splat(1.0) + self.abs())
    }
    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        unsafe { f64x8(xcopysign(self.0, rhs.0)) }
    }
}

impl VecConvertor for f64x8 {
    fn to_f64(self) -> f64x8 {
        self
    }
    fn to_i64(self) -> i64x8 {
        unsafe {
            let arr: [f64; 8] = std::mem::transmute(self.0);
            let mut result = [0i64; 8];
            for i in 0..8 {
                result[i] = arr[i] as i64;
            }
            i64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
    fn to_u64(self) -> u64x8 {
        unsafe {
            let arr: [f64; 8] = std::mem::transmute(self.0);
            let mut result = [0u64; 8];
            for i in 0..8 {
                result[i] = arr[i] as u64;
            }
            u64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_isize(self) -> isizex8 {
        self.to_i64().to_isize()
    }
    #[cfg(target_pointer_width = "64")]
    fn to_usize(self) -> usizex8 {
        self.to_u64().to_usize()
    }
}

impl Eval2 for f64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(self.0, self.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f64x8::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let i: i64x8 = unsafe { std::mem::transmute(self.0) };
        let sign_mask = i64x8::splat(-(0x8000_0000_0000_0000i64));
        let inf_mask = i64x8::splat(0x7ff0_0000_0000_0000);
        let frac_mask = i64x8::splat(0x000f_ffff_ffff_ffff);

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(i64x8::splat(0));
        let is_neg = (i & sign_mask).simd_ne(i64x8::splat(0));

        is_inf.select(
            is_neg.select(i64x8::splat(-1), i64x8::splat(1)),
            i64x8::splat(0),
        )
    }
}
