use crate::arch_simd::sleef::arch::helper_sse as helper;
use crate::{
    arch_simd::sleef::libm::sleefsimddp::{
        xacos_u1, xacosh, xasin_u1, xasinh, xatan2_u1, xatan_u1, xatanh, xcbrt_u1, xcos_u1, xcosh,
        xerf_u1, xexp, xexp10, xexp2, xexpm1, xfmax, xfmin, xhypot_u05, xlog10, xlog1p, xlog2,
        xlog_u1, xpow, xround, xsin_u1, xsincos_u1, xsinh, xsqrt_u05, xtan_u1, xtanh, xtrunc,
    },
    convertion::VecConvertor,
    simd::sleef::libm::sleefsimddp::{xceil, xcopysign, xfloor},
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};
use helper::vabs_vd_vd;

use std::arch::x86_64::*;

use crate::vectors::arch_simd::_128bit::f64x2;
use crate::vectors::arch_simd::_128bit::i64x2;
#[cfg(target_pointer_width = "64")]
use crate::vectors::arch_simd::_128bit::isizex2;
use crate::vectors::arch_simd::_128bit::u64x2;
#[cfg(target_pointer_width = "64")]
use crate::vectors::arch_simd::_128bit::usizex2;

impl PartialEq for f64x2 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_pd(self.0, other.0);
            _mm_movemask_pd(cmp) == -1
        }
    }
}

impl Default for f64x2 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { f64x2(_mm_setzero_pd()) }
    }
}

impl VecTrait<f64> for f64x2 {
    const SIZE: usize = 2;
    type Base = f64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
        unsafe {
            f64x2(_mm_add_pd(_mm_mul_pd(self.0, a.0), b.0))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        unsafe {
            f64x2(_mm_fmadd_pd(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        unsafe { _mm_cvtsd_f64(_mm_hadd_pd(self.0, self.0)) }
    }
    #[inline(always)]
    fn splat(val: f64) -> f64x2 {
        unsafe { f64x2(_mm_set1_pd(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const f64) -> Self {
        unsafe { f64x2(_mm_loadu_pd(ptr)) }
    }
}

impl f64x2 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [f64; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// check if the vector is nan
    #[inline(always)]
    pub fn is_nan(&self) -> f64x2 {
        unsafe { f64x2(_mm_cmpunord_pd(self.0, self.0)) }
    }
}

impl SimdCompare for f64x2 {
    type SimdMask = i64x2;

    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm_cmpeq_pd(self.0, other.0);
            let mask = _mm_movemask_pd(cmp);
            i64x2(_mm_set1_epi64x(if mask == 3 { -1 } else { 0 }))
        }
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm_cmpneq_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm_cmplt_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm_cmple_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
    }

    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm_cmpgt_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm_cmpge_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
    }
}

impl SimdSelect<f64x2> for i64x2 {
    #[inline(always)]
    fn select(&self, true_val: f64x2, false_val: f64x2) -> f64x2 {
        unsafe {
            f64x2(_mm_blendv_pd(
                false_val.0,
                true_val.0,
                std::mem::transmute(self.0),
            ))
        }
    }
}

impl std::ops::Add for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_add_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_sub_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_mul_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Div for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_div_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Rem for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let x: [f64; 2] = std::mem::transmute(self.0);
            let y: [f64; 2] = std::mem::transmute(rhs.0);
            let result = [x[0] % y[0], x[1] % y[1]];
            f64x2(_mm_loadu_pd(result.as_ptr()))
        }
    }
}
impl std::ops::Neg for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f64x2(_mm_xor_pd(self.0, _mm_set1_pd(-0.0))) }
    }
}

impl SimdMath<f64> for f64x2 {
    #[inline(always)]
    fn sin(self) -> Self {
        f64x2(unsafe { xsin_u1(self.0) })
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f64x2(unsafe { xcos_u1(self.0) })
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f64x2(unsafe { xtan_u1(self.0) })
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64x2(unsafe { xsqrt_u05(self.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64x2(unsafe { vabs_vd_vd(self.0) })
    }

    #[inline(always)]
    fn floor(self) -> Self {
        f64x2(unsafe { xfloor(self.0) })
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        f64x2(unsafe { xceil(self.0) })
    }

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f64x2(_mm_sub_pd(_mm_setzero_pd(), self.0)) }
    }

    #[inline(always)]
    fn round(self) -> Self {
        f64x2(unsafe { xround(self.0) })
    }

    #[inline(always)]
    fn signum(self) -> Self {
        unsafe {
            let zero = _mm_set1_pd(0.0);
            let ones = _mm_set1_pd(1.0);
            let neg_ones = _mm_set1_pd(-1.0);
            let gt = _mm_cmpgt_pd(self.0, zero);
            let lt = _mm_cmplt_pd(self.0, zero);
            f64x2(_mm_or_pd(_mm_and_pd(gt, ones), _mm_and_pd(lt, neg_ones)))
        }
    }

    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        f64x2(unsafe { xcopysign(self.0, rhs.0) })
    }

    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        let mask = self.simd_gt(Self::splat(0.0));
        mask.select(self, self * alpha)
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
        f64x2(unsafe { xpow(self.0, exp.0) })
    }

    #[inline(always)]
    fn asin(self) -> Self {
        f64x2(unsafe { xasin_u1(self.0) })
    }

    #[inline(always)]
    fn acos(self) -> Self {
        f64x2(unsafe { xacos_u1(self.0) })
    }

    #[inline(always)]
    fn atan(self) -> Self {
        f64x2(unsafe { xatan_u1(self.0) })
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        f64x2(unsafe { xsinh(self.0) })
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        f64x2(unsafe { xcosh(self.0) })
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        f64x2(unsafe { xtanh(self.0) })
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        f64x2(unsafe { xasinh(self.0) })
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        f64x2(unsafe { xacosh(self.0) })
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        f64x2(unsafe { xatanh(self.0) })
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        f64x2(unsafe { xexp2(self.0) })
    }

    #[inline(always)]
    fn exp10(self) -> Self {
        f64x2(unsafe { xexp10(self.0) })
    }

    #[inline(always)]
    fn expm1(self) -> Self {
        f64x2(unsafe { xexpm1(self.0) })
    }

    #[inline(always)]
    fn log10(self) -> Self {
        f64x2(unsafe { xlog10(self.0) })
    }

    #[inline(always)]
    fn log2(self) -> Self {
        f64x2(unsafe { xlog2(self.0) })
    }

    #[inline(always)]
    fn log1p(self) -> Self {
        f64x2(unsafe { xlog1p(self.0) })
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        f64x2(unsafe { xhypot_u05(self.0, other.0) })
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        f64x2(unsafe { xtrunc(self.0) })
    }

    #[inline(always)]
    fn erf(self) -> Self {
        f64x2(unsafe { xerf_u1(self.0) })
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        f64x2(unsafe { xcbrt_u1(self.0) })
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f64x2(unsafe { xexp(self.0) })
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f64x2(unsafe { xlog_u1(self.0) })
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f64x2(unsafe { xatan2_u1(self.0, other.0) })
    }

    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincos_u1(self.0) };
        (f64x2(ret.x), f64x2(ret.y))
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64x2(unsafe { xfmin(self.0, other.0) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64x2(unsafe { xfmax(self.0, other.0) })
    }

    #[inline(always)]
    fn recip(self) -> Self {
        unsafe { f64x2(_mm_div_pd(_mm_set1_pd(1.0), self.0)) }
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
        mask.select(self, alpha * self.expm1())
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
    fn sigmoid(self) -> Self {
        Self::splat(1.0) / (Self::splat(1.0) + (-self).exp())
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        self / (Self::splat(1.0) + self.abs())
    }
}

impl VecConvertor for f64x2 {
    #[inline(always)]
    fn to_f64(self) -> f64x2 {
        self
    }
    #[inline(always)]
    fn to_i64(self) -> i64x2 {
        unsafe {
            let arr: [f64; 2] = std::mem::transmute(self.0);
            let mut result = [0i64; 2];
            for i in 0..2 {
                result[i] = arr[i] as i64;
            }
            i64x2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
    #[inline(always)]
    fn to_u64(self) -> u64x2 {
        unsafe {
            let arr: [f64; 2] = std::mem::transmute(self.0);
            let mut result = [0u64; 2];
            for i in 0..2 {
                result[i] = arr[i] as u64;
            }
            u64x2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_isize(self) -> isizex2 {
        self.to_i64().to_isize()
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_usize(self) -> usizex2 {
        self.to_u64().to_usize()
    }
}
