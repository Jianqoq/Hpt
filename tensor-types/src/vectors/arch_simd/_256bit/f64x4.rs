use crate::{
    arch_simd::sleef::{
        arch::helper::vabs_vd_vd,
        libm::sleefsimddp::{
            xacos_u1, xacosh, xasin_u1, xasinh, xatan2_u1, xatan_u1, xatanh, xcbrt_u1, xcos_u1, xcosh, xerf_u1, xexp, xexp10, xexp2, xexpm1, xhypot_u05, xlog10, xlog1p, xlog2, xlog_u1, xpow, xround, xsin_u1, xsincos_u1, xsinh, xsqrt_u05, xtan_u1, xtanh, xtrunc
        },
    },
    traits::SimdMath,
    vectors::traits::VecTrait,
};
use std::arch::x86_64::*;
/// a vector of 4 f64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f64x4(pub(crate) __m256d);

impl Default for f64x4 {
    fn default() -> Self {
        unsafe { f64x4(_mm256_setzero_pd()) }
    }
}

impl PartialEq for f64x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_pd::<0>(self.0, other.0);
            _mm256_movemask_pd(cmp) == -1
        }
    }
}

impl VecTrait<f64> for f64x4 {
    const SIZE: usize = 4;
    type Base = f64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { f64x4(_mm256_fmadd_pd(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        unsafe {
            _mm256_storeu_pd(self.as_mut_ptr(), _mm256_loadu_pd(slice.as_ptr()));
        }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        unsafe {
            let hadd1 = _mm256_hadd_pd(self.0, self.0);
            let low = _mm256_castpd256_pd128(hadd1);
            let high = _mm256_extractf128_pd(hadd1, 1);
            let sum128 = _mm_add_pd(low, high);
            _mm_cvtsd_f64(sum128)
        }
    }
    fn splat(val: f64) -> f64x4 {
        f64x4(unsafe { _mm256_set1_pd(val) })
    }
}

impl std::ops::Add for f64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_add_pd(self.0, rhs.0) })
    }
}
impl std::ops::Sub for f64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_sub_pd(self.0, rhs.0) })
    }
}
impl std::ops::Mul for f64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_mul_pd(self.0, rhs.0) })
    }
}
impl std::ops::Div for f64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_div_pd(self.0, rhs.0) })
    }
}
impl std::ops::Rem for f64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let div = _mm256_div_pd(self.0, rhs.0);
            let floor = _mm256_floor_pd(div);
            let mul = _mm256_mul_pd(floor, rhs.0);
            f64x4(_mm256_sub_pd(self.0, mul))
        }
    }
}
impl std::ops::Neg for f64x4 {
    type Output = Self;
    fn neg(self) -> Self {
        f64x4(unsafe { _mm256_sub_pd(_mm256_setzero_pd(), self.0) })
    }
}

impl SimdMath<f64> for f64x4 {
    fn sin(self) -> Self {
        f64x4(unsafe { xsin_u1(self.0) })
    }
    fn cos(self) -> Self {
        f64x4(unsafe { xcos_u1(self.0) })
    }
    fn tan(self) -> Self {
        f64x4(unsafe { xtan_u1(self.0) })
    }

    fn square(self) -> Self {
        f64x4(unsafe { _mm256_mul_pd(self.0, self.0) })
    }

    fn sqrt(self) -> Self {
        f64x4(unsafe { xsqrt_u05(self.0) })
    }

    fn abs(self) -> Self {
        f64x4(unsafe { vabs_vd_vd(self.0) })
    }

    fn floor(self) -> Self {
        f64x4(unsafe { _mm256_floor_pd(self.0) })
    }

    fn ceil(self) -> Self {
        f64x4(unsafe { _mm256_ceil_pd(self.0) })
    }

    fn neg(self) -> Self {
        f64x4(unsafe { _mm256_sub_pd(_mm256_setzero_pd(), self.0) })
    }

    fn round(self) -> Self {
        f64x4(unsafe { xround(self.0) })
    }

    fn sign(self) -> Self {
        f64x4(unsafe { _mm256_and_pd(self.0, _mm256_set1_pd(0.0f64)) })
    }

    fn leaky_relu(self, _: f64) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        f64x4(unsafe { _mm256_max_pd(self.0, _mm256_setzero_pd()) })
    }

    fn relu6(self) -> Self {
        f64x4(unsafe { _mm256_min_pd(self.relu().0, _mm256_set1_pd(6.0f64)) })
    }

    fn pow(self, exp: Self) -> Self {
        f64x4(unsafe { xpow(self.0, exp.0) })
    }

    fn asin(self) -> Self {
        f64x4(unsafe { xasin_u1(self.0) })
    }

    fn acos(self) -> Self {
        f64x4(unsafe { xacos_u1(self.0) })
    }

    fn atan(self) -> Self {
        f64x4(unsafe { xatan_u1(self.0) })
    }

    fn sinh(self) -> Self {
        f64x4(unsafe { xsinh(self.0) })
    }

    fn cosh(self) -> Self {
        f64x4(unsafe { xcosh(self.0) })
    }

    fn tanh(self) -> Self {
        f64x4(unsafe { xtanh(self.0) })
    }

    fn asinh(self) -> Self {
        f64x4(unsafe { xasinh(self.0) })
    }

    fn acosh(self) -> Self {
        f64x4(unsafe { xacosh(self.0) })
    }

    fn atanh(self) -> Self {
        f64x4(unsafe { xatanh(self.0) })
    }

    fn exp2(self) -> Self {
        f64x4(unsafe { xexp2(self.0) })
    }

    fn exp10(self) -> Self {
        f64x4(unsafe { xexp10(self.0) })
    }

    fn expm1(self) -> Self {
        f64x4(unsafe { xexpm1(self.0) })
    }

    fn log10(self) -> Self {
        f64x4(unsafe { xlog10(self.0) })
    }

    fn log2(self) -> Self {
        f64x4(unsafe { xlog2(self.0) })
    }

    fn log1p(self) -> Self {
        f64x4(unsafe { xlog1p(self.0) })
    }

    fn hypot(self, other: Self) -> Self {
        f64x4(unsafe { xhypot_u05(self.0, other.0) })
    }

    fn trunc(self) -> Self {
        f64x4(unsafe { xtrunc(self.0) })
    }

    fn erf(self) -> Self {
        f64x4(unsafe { xerf_u1(self.0) })
    }

    fn cbrt(self) -> Self {
        f64x4(unsafe { xcbrt_u1(self.0) })
    }

    fn exp(self) -> Self {
        f64x4(unsafe { xexp(self.0) })
    }

    fn ln(self) -> Self {
        f64x4(unsafe { xlog_u1(self.0) })
    }

    fn log(self) -> Self {
        f64x4(unsafe { xlog_u1(self.0) })
    }

    fn atan2(self, other: Self) -> Self {
        f64x4(unsafe { xatan2_u1(self.0, other.0) })
    }

    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincos_u1(self.0) };
        (f64x4(ret.x), f64x4(ret.y))
    }
}
