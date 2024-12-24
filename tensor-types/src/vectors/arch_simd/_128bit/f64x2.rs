#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::arch_simd::sleef::arch::helper_aarch64 as helper;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::arch_simd::sleef::arch::helper_avx2 as helper;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "avx2")
))]
use crate::arch_simd::sleef::arch::helper_sse as helper;
use crate::{
    arch_simd::sleef::libm::sleefsimddp::{
        xacos_u1, xacosh, xasin_u1, xasinh, xatan2_u1, xatan_u1, xatanh, xcbrt_u1, xcos_u1, xcosh,
        xerf_u1, xexp, xexp10, xexp2, xexpm1, xfmax, xfmin, xhypot_u05, xlog10, xlog1p, xlog2,
        xlog_u1, xpow, xround, xsin_u1, xsincos_u1, xsinh, xsqrt_u05, xtan_u1, xtanh, xtrunc,
    },
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};
use helper::vabs_vd_vd;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_pd(self.0, other.0);
            _mm_movemask_pd(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_f64(self.0, other.0);
            vgetq_lane_u64(cmp, 0) == u64::MAX && vgetq_lane_u64(cmp, 1) == u64::MAX
        }
    }
}

impl Default for f64x2 {
    #[inline(always)]
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_setzero_pd())
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vdupq_n_f64(0.0))
        }
    }
}

impl VecTrait<f64> for f64x2 {
    const SIZE: usize = 2;
    type Base = f64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_pd(
                &mut self.0 as *mut _ as *mut f64,
                _mm_loadu_pd(slice.as_ptr()),
            );
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vld1q_f64(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_fmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vfmaq_f64(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_cvtsd_f64(_mm_hadd_pd(self.0, self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vaddvq_f64(self.0)
        }
    }
    #[inline(always)]
    fn splat(val: f64) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_set1_pd(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vdupq_n_f64(val))
        }
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
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_cmpunord_pd(self.0, self.0))
        }
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
    #[inline(always)]
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
            f64x2(vreinterpretq_f64_u64(vceqq_f64(
                abs,
                vdupq_n_f64(f64::INFINITY),
            )))
        }
    }
    /// reciprocal of the vector
    #[inline(always)]
    pub fn recip(&self) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_div_pd(_mm_set1_pd(1.0), self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use crate::simd::sleef::arch::helper_aarch64::vrec_vd_vd;
            f64x2(vrec_vd_vd(self.0))
        }
    }
}

impl SimdCompare for f64x2 {
    type SimdMask = i64x2;

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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
    #[inline(always)]
    fn select(&self, true_val: f64x2, false_val: f64x2) -> f64x2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_blendv_pd(
                false_val.0,
                true_val.0,
                std::mem::transmute(self.0),
            ))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vbslq_f64(
                vreinterpretq_u64_s64(self.0),
                true_val.0,
                false_val.0,
            ))
        }
    }
}

impl std::ops::Add for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_add_pd(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vaddq_f64(self.0, rhs.0))
        }
    }
}
impl std::ops::Sub for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_sub_pd(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vsubq_f64(self.0, rhs.0))
        }
    }
}
impl std::ops::Mul for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_mul_pd(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vmulq_f64(self.0, rhs.0))
        }
    }
}
impl std::ops::Div for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_div_pd(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vdivq_f64(self.0, rhs.0))
        }
    }
}
impl std::ops::Rem for f64x2 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
    fn neg(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_xor_pd(self.0, _mm_set1_pd(-0.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vnegq_f64(self.0))
        }
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
    fn square(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_mul_pd(self.0, self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vmulq_f64(self.0, self.0))
        }
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
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_floor_pd(self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vrndmq_f64(self.0))
        }
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_ceil_pd(self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vrndpq_f64(self.0))
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_sub_pd(_mm_setzero_pd(), self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f64x2(vnegq_f64(self.0))
        }
    }

    #[inline(always)]
    fn round(self) -> Self {
        f64x2(unsafe { xround(self.0) })
    }

    #[inline(always)]
    fn sign(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_and_pd(self.0, _mm_set1_pd(0.0f64)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let zero = vdupq_n_f64(0.0);
            let ones = vdupq_n_f64(1.0);
            let neg_ones = vdupq_n_f64(-1.0);
            let gt = vcgtq_f64(self.0, zero);
            let lt = vcltq_f64(self.0, zero);
            f64x2(vbslq_f64(gt, ones, vbslq_f64(lt, neg_ones, zero)))
        }
    }

    #[inline(always)]
    fn leaky_relu(self, _: f64) -> Self {
        todo!()
    }

    #[inline(always)]
    fn relu(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_max_pd(self.0, _mm_setzero_pd()))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let zero = vdupq_n_f64(0.0);
            f64x2(vmaxq_f64(self.0, zero))
        }
    }

    #[inline(always)]
    fn relu6(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f64x2(_mm_min_pd(self.relu().0, _mm_set1_pd(6.0f64)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let six = vdupq_n_f64(6.0);
            f64x2(vminq_f64(self.relu().0, six))
        }
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
    fn log(self) -> Self {
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
}

impl VecConvertor for f64x2 {
    #[inline(always)]
    fn to_f64(self) -> f64x2 {
        self
    }
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn to_isize(self) -> super::isizex2::isizex2 {
        self.to_i64().to_isize()
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_usize(self) -> super::usizex2::usizex2 {
        self.to_u64().to_usize()
    }
}

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    use crate::arch_simd::sleef::common::misc::SQRT_FLT_MAX;

    use super::*;
    use rug::Assign;
    use rug::Float;
    const TEST_REPEAT: usize = 100_000;
    const PRECF64: u32 = 128;
    pub fn f32_count_ulp(d: f64, c: &Float) -> f64 {
        let c2 = c.to_f64();

        if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
            return 0.;
        }

        if (c2 == 0.) && (d != 0.) {
            return 10000.;
        }

        if c2.is_infinite() && d.is_infinite() {
            return 0.;
        }

        let prec = c.prec();

        let mut fry = Float::with_val(prec, d);

        let mut frw = Float::new(prec);

        let (_, e) = c.to_f64_exp();

        frw.assign(Float::u_exp(1, e - 53_i32));

        fry -= c;
        fry /= &frw;
        let u = sleef::Sleef::abs(fry.to_f64());

        u
    }
    fn f32_gen_input(
        rng: &mut rand::rngs::ThreadRng,
        range: core::ops::RangeInclusive<f64>,
    ) -> f64 {
        use rand::Rng;
        let mut start = *range.start();
        if start == f64::MIN {
            start = -1e306;
        }
        let mut end = *range.end();
        if end == f64::MAX {
            end = 1e306;
        }
        rng.gen_range(start..=end)
    }
    fn gen_input(
        rng: &mut rand::rngs::ThreadRng,
        range: core::ops::RangeInclusive<f64>,
    ) -> float64x2_t {
        let mut arr = [0.; 2];
        for i in 0..2 {
            arr[i] = f32_gen_input(rng, range.clone());
        }
        unsafe { std::mem::transmute(arr) }
    }
    pub fn test_f_f(
        f_tested: fn(float64x2_t) -> float64x2_t,
        f_sample: fn(rug::Float) -> rug::Float,
        range: core::ops::RangeInclusive<f64>,
        ulp_ex: f64,
        name: &str,
    ) {
        test_c_f_f(
            f_tested,
            f_sample,
            range,
            |ulp, _, _| (ulp <= ulp_ex, format!("ULP: {ulp} > {ulp_ex}")),
            name,
        )
    }

    pub fn test_c_f_f(
        f_tested: fn(float64x2_t) -> float64x2_t,
        f_sample: fn(rug::Float) -> rug::Float,
        range: core::ops::RangeInclusive<f64>,
        cf: impl Fn(f64, f64, &rug::Float) -> (bool, String),
        name: &str,
    ) {
        let mut rng = rand::thread_rng();
        for n in 0..TEST_REPEAT {
            let input = gen_input(&mut rng, range.clone());
            let in_fx: [f64; 2] = unsafe { std::mem::transmute(input) };
            let out_fx: [f64; 2] = unsafe { std::mem::transmute(f_tested(input)) };
            for i in 0..2 {
                let input = in_fx[i];
                let output = out_fx[i];
                let expected = f_sample(rug::Float::with_val(PRECF64, input));
                if expected.is_nan() && output.is_nan() {
                    continue;
                }
                let ulp = f32_count_ulp(output, &expected);
                let (b, fault_string) = cf(ulp, output, &expected);
                assert!(
                    b,
                    "{}: Iteration: {n}, Position: {i}, Input: {input:e}, Output: {output}, Expected: {expected}, {}",
                    name,
                    fault_string
                );
            }
        }
    }
    #[test]
    fn tests() {
        macro_rules! define_func {
            ($func:ident, $f:ident, $x_func:expr, $range:expr) => {
                fn $func(d: float64x2_t) -> float64x2_t {
                    unsafe { $x_func(d).into() }
                }
                test_f_f($func, rug::Float::$f, $range, 1., stringify!($func));
            };
        }
        define_func!(sinf, sin, xsin_u1, f64::MIN..=f64::MAX);
        define_func!(cosf, cos, xcos_u1, f64::MIN..=f64::MAX);
        define_func!(tanf, tan, xtan_u1, f64::MIN..=f64::MAX);
        define_func!(asin, asin, xasin_u1, f64::MIN..=f64::MAX);
        define_func!(acos, acos, xacos_u1, f64::MIN..=f64::MAX);
        define_func!(atan, atan, xatan_u1, f64::MIN..=f64::MAX);
        define_func!(sinh, sinh, xsinh, -709.0..=709.0);
        define_func!(cosh, cosh, xcosh, -709.0..=709.0);
        define_func!(tanh, tanh, xtanh, -19.0..=19.0);
        define_func!(
            asinh,
            asinh,
            xasinh,
            -SQRT_FLT_MAX as f64..=SQRT_FLT_MAX as f64
        );
        define_func!(
            acosh,
            acosh,
            xacosh,
            -SQRT_FLT_MAX as f64..=SQRT_FLT_MAX as f64
        );
        define_func!(atanh, atanh, xatanh, f64::MIN..=f64::MAX);
        define_func!(round, round, xround, f64::MIN..=f64::MAX);
        define_func!(sqrt, sqrt, xsqrt_u05, f64::MIN..=f64::MAX);
        define_func!(exp, exp, xexp, -1000.0..=710.0);
        define_func!(exp2, exp2, xexp2, -2000.0..=1024.0);
        define_func!(exp10, exp10, xexp10, -350.0..=308.26);
        define_func!(expm1, exp_m1, xexpm1, -37.0..=710.0);
        define_func!(log10, log10, xlog10, 0.0..=f64::MAX);
        define_func!(log2, log2, xlog2, 0.0..=f64::MAX);
        define_func!(log1p, ln_1p, xlog1p, -1.0..=1e+38);
        define_func!(trunc, trunc, xtrunc, f64::MIN..=f64::MAX);
        define_func!(erf, erf, xerf_u1, f64::MIN..=f64::MAX);
        define_func!(cbrt, cbrt, xcbrt_u1, f64::MIN..=f64::MAX);
        define_func!(ln, ln, xlog_u1, 0.0..=f64::MAX);
    }
}
