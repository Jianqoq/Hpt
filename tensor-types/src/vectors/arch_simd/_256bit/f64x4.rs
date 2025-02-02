use crate::{
    arch_simd::sleef::{
        arch::helper_avx2::vabs_vd_vd,
        libm::sleefsimddp::{
            xacos_u1, xacosh, xasin_u1, xasinh, xatan2_u1, xatan_u1, xatanh, xcbrt_u1, xcos_u1,
            xcosh, xerf_u1, xexp, xexp10, xexp2, xexpm1, xfmax, xfmin, xhypot_u05, xlog10, xlog1p,
            xlog2, xlog_u1, xpow, xround, xsin_u1, xsincos_u1, xsinh, xsqrt_u05, xtan_u1, xtanh,
            xtrunc,
        },
    },
    convertion::VecConvertor,
    simd::sleef::{
        arch::helper_avx2::vmul_vd_vd_vd,
        libm::sleefsimddp::{xceil, xfloor},
    },
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i64x4::i64x4;

/// a vector of 2 f64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f64x4(pub(crate) __m256d);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type f64_promote = f64x4;

impl PartialEq for f64x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_EQ_OQ);
            _mm256_movemask_pd(cmp) == -1
        }
    }
}

impl Default for f64x4 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { f64x4(_mm256_setzero_pd()) }
    }
}

impl VecTrait<f64> for f64x4 {
    const SIZE: usize = 4;
    type Base = f64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        unsafe {
            _mm256_storeu_pd(
                &mut self.0 as *mut _ as *mut f64,
                _mm256_loadu_pd(slice.as_ptr()),
            );
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { f64x4(_mm256_fmadd_pd(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        unsafe { _mm256_cvtsd_f64(_mm256_hadd_pd(self.0, self.0)) }
    }
    #[inline(always)]
    fn splat(val: f64) -> f64x4 {
        unsafe { f64x4(_mm256_set1_pd(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const f64) -> Self {
        f64x4(_mm256_loadu_pd(ptr))
    }
}

impl f64x4 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [f64; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for f64x4 {
    type SimdMask = i64x4;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_EQ_OQ);
            let mask = _mm256_movemask_pd(cmp);
            i64x4(_mm256_set1_epi64x(if mask == 15 { -1 } else { 0 }))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_NEQ_OQ);
            i64x4(_mm256_castpd_si256(cmp))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_LT_OQ);
            i64x4(_mm256_castpd_si256(cmp))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_LE_OQ);
            i64x4(_mm256_castpd_si256(cmp))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_GT_OQ);
            i64x4(_mm256_castpd_si256(cmp))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_GE_OQ);
            i64x4(_mm256_castpd_si256(cmp))
        }
    }
}

impl SimdSelect<f64x4> for i64x4 {
    #[inline(always)]
    fn select(&self, true_val: f64x4, false_val: f64x4) -> f64x4 {
        unsafe {
            f64x4(_mm256_blendv_pd(
                false_val.0,
                true_val.0,
                std::mem::transmute(self.0),
            ))
        }
    }
}

impl std::ops::Add for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f64x4(_mm256_add_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { f64x4(_mm256_sub_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f64x4(_mm256_mul_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Div for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { f64x4(_mm256_div_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Rem for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let x: [f64; 4] = std::mem::transmute(self.0);
            let y: [f64; 4] = std::mem::transmute(rhs.0);
            let result = [x[0] % y[0], x[1] % y[1], x[2] % y[2], x[3] % y[3]];
            f64x4(_mm256_loadu_pd(result.as_ptr()))
        }
    }
}
impl std::ops::Neg for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f64x4(_mm256_xor_pd(self.0, _mm256_set1_pd(-0.0))) }
    }
}

impl SimdMath<f64> for f64x4 {
    #[inline(always)]
    fn sin(self) -> Self {
        f64x4(unsafe { xsin_u1(self.0) })
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f64x4(unsafe { xcos_u1(self.0) })
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f64x4(unsafe { xtan_u1(self.0) })
    }
    #[inline(always)]
    fn square(self) -> Self {
        f64x4(unsafe { vmul_vd_vd_vd(self.0, self.0) })
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64x4(unsafe { xsqrt_u05(self.0) })
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64x4(unsafe { vabs_vd_vd(self.0) })
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f64x4(unsafe { xfloor(self.0) })
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f64x4(unsafe { xceil(self.0) })
    }
    #[inline(always)]
    fn neg(self) -> Self {
        f64x4(unsafe { _mm256_sub_pd(_mm256_setzero_pd(), self.0) })
    }
    #[inline(always)]
    fn round(self) -> Self {
        f64x4(unsafe { xround(self.0) })
    }
    #[inline(always)]
    fn signum(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let zero = _mm256_set1_pd(0.0);
            let ones = _mm256_set1_pd(1.0);
            let neg_ones = _mm256_set1_pd(-1.0);
            let gt = _mm256_cmp_pd(self.0, zero, _CMP_GT_OQ);
            let lt = _mm256_cmp_pd(self.0, zero, _CMP_LT_OQ);
            f64x4(_mm256_or_pd(
                _mm256_and_pd(gt, ones),
                _mm256_and_pd(lt, neg_ones),
            ))
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
        f64x4(unsafe { xpow(self.0, exp.0) })
    }
    #[inline(always)]
    fn asin(self) -> Self {
        f64x4(unsafe { xasin_u1(self.0) })
    }
    #[inline(always)]
    fn acos(self) -> Self {
        f64x4(unsafe { xacos_u1(self.0) })
    }
    #[inline(always)]
    fn atan(self) -> Self {
        f64x4(unsafe { xatan_u1(self.0) })
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        f64x4(unsafe { xsinh(self.0) })
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        f64x4(unsafe { xcosh(self.0) })
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        f64x4(unsafe { xtanh(self.0) })
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        f64x4(unsafe { xasinh(self.0) })
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        f64x4(unsafe { xacosh(self.0) })
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        f64x4(unsafe { xatanh(self.0) })
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        f64x4(unsafe { xexp2(self.0) })
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        f64x4(unsafe { xexp10(self.0) })
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        f64x4(unsafe { xexpm1(self.0) })
    }
    #[inline(always)]
    fn log10(self) -> Self {
        f64x4(unsafe { xlog10(self.0) })
    }
    #[inline(always)]
    fn log2(self) -> Self {
        f64x4(unsafe { xlog2(self.0) })
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        f64x4(unsafe { xlog1p(self.0) })
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        f64x4(unsafe { xhypot_u05(self.0, other.0) })
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        f64x4(unsafe { xtrunc(self.0) })
    }
    #[inline(always)]
    fn erf(self) -> Self {
        f64x4(unsafe { xerf_u1(self.0) })
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        f64x4(unsafe { xcbrt_u1(self.0) })
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64x4(unsafe { xexp(self.0) })
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f64x4(unsafe { xlog_u1(self.0) })
    }
    #[inline(always)]
    fn log(self) -> Self {
        f64x4(unsafe { xlog_u1(self.0) })
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f64x4(unsafe { xatan2_u1(self.0, other.0) })
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincos_u1(self.0) };
        (f64x4(ret.x), f64x4(ret.y))
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64x4(unsafe { xfmin(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64x4(unsafe { xfmax(self.0, other.0) })
    }
    
    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let sixth = f64x4::splat(1.0 / 6.0);
        let half = f64x4::splat(0.5);
        let one = f64x4::splat(1.0);
        let zero = f64x4::splat(0.0);
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
        Self(unsafe {
            let is_nan = _mm256_cmp_pd(self.0, self.0, _CMP_UNORD_Q);
            let is_zero = _mm256_cmp_pd(self.0, _mm256_setzero_pd(), _CMP_EQ_OQ);
            let recip = _mm256_div_pd(_mm256_set1_pd(1.0), self.0);
            _mm256_blendv_pd(
                recip,
                _mm256_or_pd(
                    _mm256_and_pd(is_zero, _mm256_set1_pd(f64::INFINITY)),
                    _mm256_and_pd(is_nan, _mm256_set1_pd(f64::NAN)),
                ),
                _mm256_or_pd(is_nan, is_zero),
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

impl VecConvertor for f64x4 {
    fn to_f64(self) -> f64x4 {
        self
    }
    fn to_i64(self) -> super::i64x4::i64x4 {
        unsafe {
            let arr: [f64; 4] = std::mem::transmute(self.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = arr[i] as i64;
            }
            super::i64x4::i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    fn to_u64(self) -> super::u64x4::u64x4 {
        unsafe {
            let arr: [f64; 4] = std::mem::transmute(self.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = arr[i] as u64;
            }
            super::u64x4::u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_isize(self) -> super::isizex4::isizex4 {
        self.to_i64().to_isize()
    }
    #[cfg(target_pointer_width = "64")]
    fn to_usize(self) -> super::usizex4::usizex4 {
        self.to_u64().to_usize()
    }
}

impl FloatOutBinary2 for f64x4 {
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
        ];
        f64x4(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for f64x4 {
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
    fn __clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for f64x4 {
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
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(f64x4::splat(0.0)) + alpha * self.min(f64x4::splat(0.0))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self.trunc()
    }
}

impl Eval2 for f64x4 {
    type Output = i64x4;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe {
            i64x4(std::mem::transmute(_mm256_cmp_pd(
                self.0,
                self.0,
                _CMP_UNORD_Q,
            )))
        }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f64x4::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let i: i64x4 = unsafe { std::mem::transmute(self.0) };
        let sign_mask = i64x4::splat(-(0x8000_0000_0000_0000i64));
        let inf_mask = i64x4::splat(0x7ff0_0000_0000_0000);
        let frac_mask = i64x4::splat(0x000f_ffff_ffff_ffff);

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(i64x4::splat(0));
        let is_neg = (i & sign_mask).simd_ne(i64x4::splat(0));

        is_inf.select(
            is_neg.select(i64x4::splat(-1), i64x4::splat(1)),
            i64x4::splat(0),
        )
    }
}
