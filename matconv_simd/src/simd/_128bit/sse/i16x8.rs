use crate::{
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOutUnary2},
};

use std::arch::x86_64::*;

use crate::vectors::arch_simd::_128bit::i16x8;

impl PartialEq for i16x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi16(self.0, other.0);
            _mm_movemask_epi8(cmp) == 0xffff
        }
    }
}
impl Default for i16x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i16x8(_mm_setzero_si128()) }
    }
}
impl VecTrait<i16> for i16x8 {
    const SIZE: usize = 8;
    type Base = i16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        unsafe {
            _mm_storeu_si128(
                &mut self.0,
                _mm_loadu_si128(slice.as_ptr() as *const __m128i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i16x8(_mm_add_epi16(self.0, _mm_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> i16 {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: i16) -> i16x8 {
        unsafe { i16x8(_mm_set1_epi16(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i16) -> Self {
        unsafe { i16x8(_mm_loadu_si128(ptr as *const __m128i)) }
    }
}

impl SimdCompare for i16x8 {
    type SimdMask = i16x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x8 {
        unsafe { i16x8(_mm_cmpeq_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x8 {
        unsafe {
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            i16x8(_mm_xor_si128(eq, _mm_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x8 {
        unsafe { i16x8(_mm_cmplt_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x8 {
        unsafe {
            let lt = _mm_cmplt_epi16(self.0, other.0);
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            i16x8(_mm_or_si128(lt, eq))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x8 {
        unsafe { i16x8(_mm_cmpgt_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x8 {
        unsafe {
            let gt = _mm_cmpgt_epi16(self.0, other.0);
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            i16x8(_mm_or_si128(gt, eq))
        }
    }
}

impl SimdSelect<i16x8> for crate::vectors::arch_simd::_128bit::i16x8 {
    #[inline(always)]
    fn select(&self, true_val: i16x8, false_val: i16x8) -> i16x8 {
        unsafe { i16x8(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            let arr2: [i16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            i16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            let arr2: [i16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            i16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i16x8(_mm_sign_epi16(self.0, _mm_set1_epi16(-1))) }
    }
}
impl std::ops::BitAnd for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i16x8(_mm_xor_si128(self.0, _mm_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 8] = std::mem::transmute(self.0);
            let b: [i16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 8] = std::mem::transmute(self.0);
            let b: [i16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl SimdMath<i16> for i16x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i16x8(_mm_max_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i16x8(_mm_min_epi16(self.0, other.0)) }
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
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [i16; 8] = std::mem::transmute(self.0);
            let b: [i16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0i16; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            i16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
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
        unsafe { i16x8(_mm_abs_epi16(self.0)) }
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
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(i16x8::splat(0)) + alpha * self.min(i16x8::splat(0))
    }
}

impl FloatOutBinary2 for i16x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i16")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for i16x8");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            let arr2: [i16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 8] = [0; 8];
            for i in 0..8 {
                if arr2[i] < 0 {
                    panic!("Power operation is not supported for negative i16");
                }
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            i16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}

impl NormalOutUnary2 for i16x8 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        unsafe { i16x8(_mm_abs_epi16(self.0)) }
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
        self.max(i16x8::splat(0)) + alpha * self.min(i16x8::splat(0))
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

impl Eval2 for i16x8 {
    type Output = i16x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i16x8::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm_cmpeq_epi16(self.0, _mm_setzero_si128());
            let result = _mm_andnot_si128(eq, _mm_set1_epi16(1));
            Self(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i16x8::default()
    }
}
