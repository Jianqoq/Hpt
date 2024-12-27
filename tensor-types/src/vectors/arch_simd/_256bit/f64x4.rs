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
    /// check if the vector is nan
    #[inline(always)]
    pub fn is_nan(&self) -> f64x4 {
        unsafe { f64x4(_mm256_cmp_pd(self.0, self.0, _CMP_UNORD_Q)) }
    }
    /// check if the vector is infinite
    #[inline(always)]
    pub fn is_infinite(&self) -> f64x4 {
        unsafe {
            let abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), self.0);
            f64x4(_mm256_cmp_pd(
                abs,
                _mm256_set1_pd(f64::INFINITY),
                _CMP_EQ_OQ,
            ))
        }
    }
    /// reciprocal of the vector
    #[inline(always)]
    pub fn recip(&self) -> f64x4 {
        unsafe { f64x4(_mm256_div_pd(_mm256_set1_pd(1.0), self.0)) }
    }
}

impl SimdCompare for f64x4 {
    type SimdMask = i64x4;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let cmp = _mm256_cmp_pd(self.0, other.0, _CMP_EQ_OQ);
            let mask = _mm256_movemask_pd(cmp);
            i64x4(_mm256_set1_epi64x(if mask == 3 { -1 } else { 0 }))
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
        f64x4(unsafe { _mm256_mul_pd(self.0, self.0) })
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
        f64x4(unsafe { _mm256_floor_pd(self.0) })
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f64x4(unsafe { _mm256_ceil_pd(self.0) })
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
    fn sign(self) -> Self {
        f64x4(unsafe { _mm256_and_pd(self.0, _mm256_set1_pd(0.0f64)) })
    }
    #[inline(always)]
    fn leaky_relu(self, _: f64) -> Self {
        todo!()
    }
    #[inline(always)]
    fn relu(self) -> Self {
        f64x4(unsafe { _mm256_max_pd(self.0, _mm256_setzero_pd()) })
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        f64x4(unsafe { _mm256_min_pd(self.relu().0, _mm256_set1_pd(6.0f64)) })
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
    fn __clip(self, min: Self, max: Self) -> Self {
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
    fn __sign(self) -> Self {
        self.sign()
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
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

impl Eval2 for f64x4 {
    type Output = i64x4;
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
    ) -> __m256d {
        let mut arr = [0.; 4];
        for i in 0..4 {
            arr[i] = f32_gen_input(rng, range.clone());
        }
        unsafe { std::mem::transmute(arr) }
    }
    pub fn test_f_f(
        f_tested: fn(__m256d) -> __m256d,
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
        f_tested: fn(__m256d) -> __m256d,
        f_sample: fn(rug::Float) -> rug::Float,
        range: core::ops::RangeInclusive<f64>,
        cf: impl Fn(f64, f64, &rug::Float) -> (bool, String),
        name: &str,
    ) {
        let mut rng = rand::thread_rng();
        for n in 0..TEST_REPEAT {
            let input = gen_input(&mut rng, range.clone());
            let in_fx: [f64; 4] = unsafe { std::mem::transmute(input) };
            let out_fx: [f64; 4] = unsafe { std::mem::transmute(f_tested(input)) };
            for i in 0..4 {
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
                fn $func(d: __m256d) -> __m256d {
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
