use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOutUnary2},
};
use std::arch::aarch64::*;

use crate::vectors::arch_simd::_128bit::f32x4;
use crate::vectors::arch_simd::_128bit::i32x4;
#[cfg(target_pointer_width = "32")]
use crate::vectors::arch_simd::_128bit::isizex2;
use crate::vectors::arch_simd::_128bit::u32x4;
#[cfg(target_pointer_width = "32")]
use crate::vectors::arch_simd::_128bit::usizex2;

impl PartialEq for i32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_s32(self.0, other.0);
            vmaxvq_u32(cmp) == 0xffffffff && vminvq_u32(cmp) == 0xffffffff
        }
    }
}

impl Default for i32x4 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i32x4(vdupq_n_s32(0)) }
    }
}

impl VecTrait<i32> for i32x4 {
    const SIZE: usize = 4;
    type Base = i32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_s32(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        unsafe {
            let sum = vaddvq_s32(self.0);
            sum as i32
        }
    }
    #[inline(always)]
    fn splat(val: i32) -> i32x4 {
        unsafe { i32x4(vdupq_n_s32(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i32) -> Self {
        unsafe { i32x4(vld1q_s32(ptr)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        Self(unsafe { vmlaq_laneq_s32::<LANE>(b.0, self.0, a.0) })
    }
}

impl SimdCompare for i32x4 {
    type SimdMask = i32x4;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i32x4 {
        unsafe {
            let cmp = vceqq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i32x4 {
        unsafe {
            let eq = vceqq_s32(self.0, other.0);
            i32x4(veorq_s32(std::mem::transmute(eq), vdupq_n_s32(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i32x4 {
        unsafe {
            let cmp = vcltq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i32x4 {
        unsafe {
            let cmp = vcleq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i32x4 {
        unsafe {
            let cmp = vcgtq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i32x4 {
        unsafe {
            let cmp = vcgeq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
}

impl SimdSelect<i32x4> for crate::vectors::arch_simd::_128bit::i32x4 {
    #[inline(always)]
    fn select(&self, true_val: i32x4, false_val: i32x4) -> i32x4 {
        unsafe {
            let zero = vdupq_n_s32(0);
            let cmp = vcltq_s32(self.0, zero);
            i32x4(vbslq_s32(cmp, true_val.0, false_val.0))
        }
    }
}

impl std::ops::Add for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vaddq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vsubq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vmulq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            #[cfg(target_feature = "neon")]
            return i32x4(vld1q_s32(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] % arr2[i];
            }
            #[cfg(target_feature = "neon")]
            return i32x4(vld1q_s32(arr3.as_ptr()));
        }
    }
}

impl std::ops::Neg for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i32x4(vnegq_s32(self.0)) }
    }
}
impl std::ops::BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vandq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vorrq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(veorq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i32x4(vmvnq_s32(self.0)) }
    }
}
impl std::ops::Shl for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            #[cfg(target_feature = "neon")]
            return i32x4(vld1q_s32(result.as_ptr()));
        }
    }
}
impl std::ops::Shr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            #[cfg(target_feature = "neon")]
            return i32x4(vld1q_s32(result.as_ptr()));
        }
    }
}
impl SimdMath<i32> for i32x4 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i32x4(vmaxq_s32(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i32x4(vminq_s32(self.0, other.0)) }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        self.max(Self::splat(0))
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        self.min(Self::splat(6)).max(Self::splat(0))
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self
    }
    #[inline(always)]
    fn round(self) -> Self {
        self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { i32x4(vabsq_s32(self.0)) }
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn signum(self) -> Self {
        let zero = Self::splat(0);
        let gt = self.simd_gt(zero);
        let lt = self.simd_lt(zero);
        let pos = gt & Self::splat(1);
        let neg = lt & Self::splat(-1);
        pos | neg
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 4];
            for i in 0..4 {
                result[i] = a[i].pow(b[i] as u32);
            }
            return i32x4(vld1q_s32(result.as_ptr()));
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(i32x4::splat(0)) + alpha * self.min(i32x4::splat(0))
    }
}

impl VecConvertor for i32x4 {
    #[inline(always)]
    fn to_i32(self) -> i32x4 {
        self
    }
    #[inline(always)]
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f32(self) -> f32x4 {
        unsafe { f32x4(vcvtq_f32_s32(self.0)) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_isize(self) -> isizex2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_usize(self) -> usizex2 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for i32x4 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i32")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for i32x4");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                if arr2[i] < 0 {
                    panic!("Power operation is not supported for negative i32");
                }
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            return i32x4(vld1q_s32(arr3.as_ptr()));
        }
    }
}

impl NormalOutUnary2 for i32x4 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        unsafe { i32x4(vabsq_s32(self.0)) }
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(i32x4::splat(0)) + alpha * self.min(i32x4::splat(0))
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
        self.abs() * rhs.signum()
    }
}

impl Eval2 for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i32x4::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let neq = vmvnq_s32(vreinterpretq_s32_u32(vceqq_s32(self.0, vdupq_n_s32(0))));
            i32x4(vandq_s32(neq, vdupq_n_s32(1)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i32x4::default()
    }
}
