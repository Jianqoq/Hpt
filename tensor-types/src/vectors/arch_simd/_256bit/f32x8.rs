use crate::{
    arch_simd::sleef::{
        arch::helper::vabs_vf_vf,
        libm::sleefsimdsp::{
            xacosf_u1, xacoshf, xasinf_u1, xasinhf, xatanf_u1, xatanhf, xcbrtf_u1, xcosf_u1,
            xcoshf, xerff_u1, xexp10f, xexp2f, xexpf, xexpm1f, xhypotf_u05, xlog10f, xlog1pf,
            xlog2f, xlogf_u1, xpowf, xroundf, xsinf_u1, xsinhf, xsqrtf_u05, xtanf_u1, xtanhf,
            xtruncf,
        },
    },
    traits::SimdMath,
    vectors::traits::{SimdSelect, VecTrait},
};
use std::arch::x86_64::*;

/// a vector of 8 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f32x8(pub(crate) __m256);

impl Default for f32x8 {
    fn default() -> Self {
        unsafe { f32x8(_mm256_setzero_ps()) }
    }
}

impl PartialEq for f32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ);
            _mm256_movemask_ps(cmp) == -1
        }
    }
}

impl VecTrait<f32> for f32x8 {
    const SIZE: usize = 8;
    type Base = f32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32x8(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.0 = unsafe { _mm256_loadu_ps(slice.as_ptr()) };
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self as *const _ as *const f32
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self as *mut _ as *mut f32
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut f32 {
        unsafe { std::mem::transmute(self.as_ptr()) }
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        unsafe {
            let hadd1 = _mm256_hadd_ps(self.0, self.0);
            let hadd2 = _mm256_hadd_ps(hadd1, hadd1);
            let low = _mm256_castps256_ps128(hadd2);
            let high = _mm256_extractf128_ps(hadd2, 1);
            let sum128 = _mm_add_ps(low, high);
            _mm_cvtss_f32(sum128)
        }
    }
    fn splat(val: f32) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(val) })
    }
}

impl SimdSelect<f32x8> for crate::vectors::arch_simd::_256bit::u32x8::u32x8 {
    fn select(&self, true_val: f32x8, false_val: f32x8) -> f32x8 {
        unsafe {
            let mask = _mm256_castsi256_ps(self.0);
            f32x8(_mm256_blendv_ps(false_val.0, true_val.0, mask))
        }
    }
}

impl std::ops::Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_add_ps(self.0, rhs.0) })
    }
}

impl std::ops::Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_sub_ps(self.0, rhs.0) })
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_mul_ps(self.0, rhs.0) })
    }
}

impl std::ops::Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_div_ps(self.0, rhs.0) })
    }
}

impl std::ops::Rem for f32x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let div = _mm256_div_ps(self.0, rhs.0);
            let floor = _mm256_floor_ps(div);
            let mul = _mm256_mul_ps(floor, rhs.0);
            f32x8(_mm256_sub_ps(self.0, mul))
        }
    }
}
impl std::ops::Neg for f32x8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        f32x8(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) })
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Assign;
    use rug::{ops::Pow, Float};
    const TEST_REPEAT: usize = 100_000;
    const PRECF32: u32 = 80;
    pub fn f32_count_ulp(d: f32, c: &Float) -> f32 {
        let c2 = c.to_f32();

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

        let (_, e) = c.to_f32_exp();

        frw.assign(Float::u_exp(1, e - 24_i32));

        fry -= c;
        fry /= &frw;
        let u = f32::from_bits(0x_7fff_ffff & fry.to_f32().to_bits());

        u
    }
    fn f32_gen_input(
        rng: &mut rand::rngs::ThreadRng,
        range: core::ops::RangeInclusive<f32>,
    ) -> f32 {
        use rand::Rng;
        let mut start = *range.start();
        if start == f32::MIN {
            start = -1e37;
        }
        let mut end = *range.end();
        if end == f32::MAX {
            end = 1e37;
        }
        rng.gen_range(start..=end)
    }
    fn gen_input<const N: usize>(
        rng: &mut rand::rngs::ThreadRng,
        range: core::ops::RangeInclusive<f32>,
    ) -> std::simd::Simd<f32, N>
    where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    {
        let mut arr = [0.; N];
        for i in 0..N {
            arr[i] = f32_gen_input(rng, range.clone());
        }
        arr.into()
    }
    pub fn test_f_f<const N: usize>(
        f_tested: fn(std::simd::Simd<f32, N>) -> std::simd::Simd<f32, N>,
        f_sample: fn(rug::Float) -> rug::Float,
        range: core::ops::RangeInclusive<f32>,
        ulp_ex: f32,
    ) where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    {
        test_c_f_f(f_tested, f_sample, range, |ulp, _, _| {
            (ulp <= ulp_ex, format!("ULP: {ulp} > {ulp_ex}"))
        })
    }

    pub fn test_c_f_f<const N: usize>(
        f_tested: fn(std::simd::Simd<f32, N>) -> std::simd::Simd<f32, N>,
        f_sample: fn(rug::Float) -> rug::Float,
        range: core::ops::RangeInclusive<f32>,
        cf: impl Fn(f32, f32, &rug::Float) -> (bool, String),
    ) where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    {
        let mut rng = rand::thread_rng();
        for n in 0..TEST_REPEAT {
            let in_fx = gen_input(&mut rng, range.clone());
            let out_fx = f_tested(in_fx);
            for i in 0..N {
                let input = in_fx[i];
                let output = out_fx[i];
                let expected = f_sample(rug::Float::with_val(PRECF32, input));
                if expected.is_nan() && output.is_nan() {
                    continue;
                }
                let ulp = f32_count_ulp(output, &expected);
                let (b, fault_string) = cf(ulp, output, &expected);
                assert!(
                    b,
                    "Iteration: {n}, Position: {i}, Input: {input:e}, Output: {output}, Expected: {expected}, {}",
                    fault_string
                );
            }
        }
    }
    #[test]
    fn test_sinf() {
        fn sinf(d: std::simd::Simd<f32, 8>) -> std::simd::Simd<f32, 8>
        where
            std::simd::LaneCount<8>: std::simd::SupportedLaneCount,
        {
            unsafe { xsinf_u1(std::mem::transmute(d)).into() }
        }
        test_f_f::<8>(sinf, rug::Float::sin, f32::MIN..=f32::MAX, 1.);
    }
}
