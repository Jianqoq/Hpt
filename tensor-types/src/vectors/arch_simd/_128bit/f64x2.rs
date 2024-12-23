#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::arch_simd::sleef::arch::helper_avx2 as helper;
#[cfg(all(target_arch = "x86_64", target_feature = "sse", not(target_feature = "avx2")))]
use crate::arch_simd::sleef::arch::helper_sse as helper;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::arch_simd::sleef::arch::helper_neon as helper;
use helper::vabs_vd_vd;
use crate::{
    arch_simd::sleef::libm::sleefsimddp::{
        xacos_u1,
        xacosh,
        xasin_u1,
        xasinh,
        xatan2_u1,
        xatan_u1,
        xatanh,
        xcbrt_u1,
        xcos_u1,
        xcosh,
        xerf_u1,
        xexp,
        xexp10,
        xexp2,
        xexpm1,
        xfmax,
        xfmin,
        xhypot_u05,
        xlog10,
        xlog1p,
        xlog2,
        xlog_u1,
        xpow,
        xround,
        xsin_u1,
        xsincos_u1,
        xsinh,
        xsqrt_u05,
        xtan_u1,
        xtanh,
        xtrunc,
    },
    convertion::VecConvertor,
    traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait },
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::i64x2::i64x2;

/// a vector of 2 f64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f64x2(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128d,
    #[cfg(target_arch = "aarch64")] pub(crate) float64x2_t,
);

impl PartialEq for f64x2 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_pd(self.0, other.0);
            _mm_movemask_pd(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_f64(self.0, other.0);
            vgetq_lane_u64(cmp, 0) == u64::MAX && 
            vgetq_lane_u64(cmp, 1) == u64::MAX
        }
    }
}

impl Default for f64x2 {
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_setzero_pd()) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vdupq_n_f64(0.0)) }
    }
}

impl VecTrait<f64> for f64x2 {
    const SIZE: usize = 2;
    type Base = f64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_pd(&mut self.0 as *mut _ as *mut f64, _mm_loadu_pd(slice.as_ptr()));
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vld1q_f64(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_fmadd_pd(self.0, a.0, b.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vfmaq_f64(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        #[cfg(target_arch = "x86_64")]
        unsafe { _mm_cvtsd_f64(_mm_hadd_pd(self.0, self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { vaddvq_f64(self.0) }
    }
    fn splat(val: f64) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_set1_pd(val)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vdupq_n_f64(val)) }
    }
}

impl f64x2 {
    #[allow(unused)]
    fn as_array(&self) -> [f64; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// check if the vector is nan
    pub fn is_nan(&self) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_cmpunord_pd(self.0, self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { 
            // vceqq_f64 returns true (all bits set) if values are equal
            // NaN compared with itself always returns false
            let cmp = vceqq_f64(self.0, self.0);
            // Invert the comparison result to get true for NaN values
            f64x2(vreinterpretq_f64_u64(vceqzq_u64(cmp)))
        }
    }
    /// check if the vector is infinite
    pub fn is_infinite(&self) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let abs = _mm_andnot_pd(_mm_set1_pd(-0.0), self.0);
            f64x2(_mm_cmpeq_pd(abs, _mm_set1_pd(f64::INFINITY)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Clear sign bit to get absolute value
            let abs = vabsq_f64(self.0);
            // Compare with infinity
            f64x2(vreinterpretq_f64_u64(vceqq_f64(abs, vdupq_n_f64(f64::INFINITY))))
        }
    }
    /// reciprocal of the vector
    pub fn recip(&self) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_div_pd(_mm_set1_pd(1.0), self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vrecpeq_f64(self.0)) }
    }
}

impl SimdCompare for f64x2 {
    type SimdMask = i64x2;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_pd(self.0, other.0);
            let mask = _mm_movemask_pd(cmp);
            i64x2(_mm_set1_epi64x(if mask == 3 { -1 } else { 0 }))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_f64(self.0, other.0);
            i64x2(vreinterpretq_s64_u64(cmp))
        }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpneq_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_f64(self.0, other.0);
            i64x2(vreinterpretq_s64_u64(vceqzq_u64(cmp)))
        }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmplt_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcltq_f64(self.0, other.0);
            i64x2(vreinterpretq_s64_u64(cmp))
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmple_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcleq_f64(self.0, other.0);
            i64x2(vreinterpretq_s64_u64(cmp))
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpgt_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcgtq_f64(self.0, other.0);
            i64x2(vreinterpretq_s64_u64(cmp))
        }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpge_pd(self.0, other.0);
            i64x2(_mm_castpd_si128(cmp))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcgeq_f64(self.0, other.0);
            i64x2(vreinterpretq_s64_u64(cmp))
        }
    }
}

impl SimdSelect<f64x2> for i64x2 {
    fn select(&self, true_val: f64x2, false_val: f64x2) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_blendv_pd(false_val.0, true_val.0, std::mem::transmute(self.0))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vbslq_f64(vreinterpretq_u64_s64(self.0), true_val.0, false_val.0)) }
    }
}

impl std::ops::Add for f64x2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_add_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vaddq_f64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for f64x2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_sub_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vsubq_f64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_mul_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vmulq_f64(self.0, rhs.0)) }
    }
}
impl std::ops::Div for f64x2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_div_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vdivq_f64(self.0, rhs.0)) }
    }
}
impl std::ops::Rem for f64x2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let x: [f64; 2] = std::mem::transmute(self.0);
            let y: [f64; 2] = std::mem::transmute(rhs.0);
            let result = [x[0] % y[0], x[1] % y[1]];
            f64x2(_mm_loadu_pd(result.as_ptr()))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let x: [f64; 2] = std::mem::transmute(self.0);
            let y: [f64; 2] = std::mem::transmute(rhs.0);
                let result = [x[0] % y[0], x[1] % y[1]];
            f64x2(vld1q_f64(result.as_ptr()))
        }
    }
}
impl std::ops::Neg for f64x2 {
    type Output = Self;
    fn neg(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_xor_pd(self.0, _mm_set1_pd(-0.0))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vnegq_f64(self.0)) }
    }
}

impl SimdMath<f64> for f64x2 {
    fn sin(self) -> Self {
        f64x2(unsafe { xsin_u1(self.0) })
    }
    fn cos(self) -> Self {
        f64x2(unsafe { xcos_u1(self.0) })
    }
    fn tan(self) -> Self {
        f64x2(unsafe { xtan_u1(self.0) })
    }

    fn square(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_mul_pd(self.0, self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vmulq_f64(self.0, self.0)) }
    }

    fn sqrt(self) -> Self {
        f64x2(unsafe { xsqrt_u05(self.0) })
    }

    fn abs(self) -> Self {
        f64x2(unsafe { vabs_vd_vd(self.0) })
    }

    fn floor(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_floor_pd(self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vrndmq_f64(self.0)) }
    }

    fn ceil(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_ceil_pd(self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vrndpq_f64(self.0)) }
    }

    fn neg(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_sub_pd(_mm_setzero_pd(), self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { f64x2(vnegq_f64(self.0)) }
    }

    fn round(self) -> Self {
        f64x2(unsafe { xround(self.0) })
    }

    fn sign(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_and_pd(self.0, _mm_set1_pd(0.0f64))) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let zero = vdupq_n_f64(0.0);
            let ones = vdupq_n_f64(1.0);
            let neg_ones = vdupq_n_f64(-1.0);
            let gt = vcgtq_f64(self.0, zero);
            let lt = vcltq_f64(self.0, zero);
            f64x2(vbslq_f64(
                gt,
                ones,
                vbslq_f64(lt, neg_ones, zero)
            ))
        }
    }

    fn leaky_relu(self, _: f64) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_max_pd(self.0, _mm_setzero_pd())) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let zero = vdupq_n_f64(0.0);
            f64x2(vmaxq_f64(self.0, zero))
        }
    }

    fn relu6(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { f64x2(_mm_min_pd(self.relu().0, _mm_set1_pd(6.0f64))) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let six = vdupq_n_f64(6.0);
            f64x2(vminq_f64(self.relu().0, six))
        }
    }

    fn pow(self, exp: Self) -> Self {
        f64x2(unsafe { xpow(self.0, exp.0) })
    }

    fn asin(self) -> Self {
        f64x2(unsafe { xasin_u1(self.0) })
    }

    fn acos(self) -> Self {
        f64x2(unsafe { xacos_u1(self.0) })
    }

    fn atan(self) -> Self {
        f64x2(unsafe { xatan_u1(self.0) })
    }

    fn sinh(self) -> Self {
        f64x2(unsafe { xsinh(self.0) })
    }

    fn cosh(self) -> Self {
        f64x2(unsafe { xcosh(self.0) })
    }

    fn tanh(self) -> Self {
        f64x2(unsafe { xtanh(self.0) })
    }

    fn asinh(self) -> Self {
        f64x2(unsafe { xasinh(self.0) })
    }

    fn acosh(self) -> Self {
        f64x2(unsafe { xacosh(self.0) })
    }

    fn atanh(self) -> Self {
        f64x2(unsafe { xatanh(self.0) })
    }

    fn exp2(self) -> Self {
        f64x2(unsafe { xexp2(self.0) })
    }

    fn exp10(self) -> Self {
        f64x2(unsafe { xexp10(self.0) })
    }

    fn expm1(self) -> Self {
        f64x2(unsafe { xexpm1(self.0) })
    }

    fn log10(self) -> Self {
        f64x2(unsafe { xlog10(self.0) })
    }

    fn log2(self) -> Self {
        f64x2(unsafe { xlog2(self.0) })
    }

    fn log1p(self) -> Self {
        f64x2(unsafe { xlog1p(self.0) })
    }

    fn hypot(self, other: Self) -> Self {
        f64x2(unsafe { xhypot_u05(self.0, other.0) })
    }

    fn trunc(self) -> Self {
        f64x2(unsafe { xtrunc(self.0) })
    }

    fn erf(self) -> Self {
        f64x2(unsafe { xerf_u1(self.0) })
    }

    fn cbrt(self) -> Self {
        f64x2(unsafe { xcbrt_u1(self.0) })
    }

    fn exp(self) -> Self {
        f64x2(unsafe { xexp(self.0) })
    }

    fn ln(self) -> Self {
        f64x2(unsafe { xlog_u1(self.0) })
    }

    fn log(self) -> Self {
        f64x2(unsafe { xlog_u1(self.0) })
    }

    fn atan2(self, other: Self) -> Self {
        f64x2(unsafe { xatan2_u1(self.0, other.0) })
    }

    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincos_u1(self.0) };
        (f64x2(ret.x), f64x2(ret.y))
    }

    fn min(self, other: Self) -> Self {
        f64x2(unsafe { xfmin(self.0, other.0) })
    }

    fn max(self, other: Self) -> Self {
        f64x2(unsafe { xfmax(self.0, other.0) })
    }
}

impl VecConvertor for f64x2 {
    fn to_f64(self) -> f64x2 {
        self
    }
    fn to_i64(self) -> super::i64x2::i64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let arr: [f64; 2] = std::mem::transmute(self.0);
            let mut result = [0i64; 2];
            for i in 0..2 {
                result[i] = arr[i] as i64;
            }
            super::i64x2::i64x2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let arr: [f64; 2] = std::mem::transmute(self.0);
            let mut result = [0i64; 2];
            for i in 0..2 {
                result[i] = arr[i] as i64;
            }
            super::i64x2::i64x2(vld1q_s64(result.as_ptr()))
        }
    }
    fn to_u64(self) -> super::u64x2::u64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let arr: [f64; 2] = std::mem::transmute(self.0);
            let mut result = [0u64; 2];
            for i in 0..2 {
                result[i] = arr[i] as u64;
            }
            super::u64x2::u64x2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let arr: [f64; 2] = std::mem::transmute(self.0);
            let mut result = [0u64; 2];
            for i in 0..2 {
                result[i] = arr[i] as u64;
            }
            super::u64x2::u64x2(vld1q_u64(result.as_ptr()))
        }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_isize(self) -> super::isizex2::isizex2 {
        self.to_i64().to_isize()
    }
    #[cfg(target_pointer_width = "64")]
    fn to_usize(self) -> super::usizex2::usizex2 {
        self.to_u64().to_usize()
    }
}
