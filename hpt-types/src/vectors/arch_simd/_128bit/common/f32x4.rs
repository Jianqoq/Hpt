use crate::traits::{SimdMath, VecTrait};
use crate::type_promote::{FloatOutBinary2, NormalOut2, NormalOutUnary2};

#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 4 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f32x4(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128,
    #[cfg(target_arch = "aarch64")] pub(crate) float32x4_t,
);

#[allow(non_camel_case_types)]
pub(crate) type f32_promote = f32x4;

impl f32x4 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl FloatOutBinary2 for f32x4 {
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
        f32x4(unsafe { std::mem::transmute(res) })
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

impl NormalOut2 for f32x4 {
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

impl NormalOutUnary2 for f32x4 {
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
        self.max(f32x4::splat(0.0)) + alpha * self.min(f32x4::splat(0.0))
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
