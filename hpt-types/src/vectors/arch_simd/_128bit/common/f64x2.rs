use crate::{
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

#[cfg(target_feature = "neon")]
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

#[allow(non_camel_case_types)]
pub(crate) type f64_promote = f64x2;

impl FloatOutBinary2 for f64x2 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [self[0].log(base[0]), self[1].log(base[1])];
        f64x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        self.hypot(rhs)
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        self.pow(rhs)
    }
}

impl NormalOut2 for f64x2 {
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

impl NormalOutUnary2 for f64x2 {
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
    fn __trunc(self) -> Self {
        self.trunc()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.leaky_relu(alpha)
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
    fn __copysign(self, rhs: Self) -> Self {
        self.copysign(rhs)
    }
}

impl Eval2 for f64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe { std::mem::transmute(self.is_nan()) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f64x2::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let i: i64x2 = unsafe { std::mem::transmute(self.0) };
        let sign_mask = i64x2::splat(-0x8000_0000_0000_0000i64);
        let inf_mask = i64x2::splat(0x7ff0_0000_0000_0000);
        let frac_mask = i64x2::splat(0x000f_ffff_ffff_ffff);

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(i64x2::splat(0));
        let is_neg = (i & sign_mask).simd_ne(i64x2::splat(0));

        is_inf.select(
            is_neg.select(i64x2::splat(-1), i64x2::splat(1)),
            i64x2::splat(0),
        )
    }
}
