use crate::{
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};
use std::arch::x86_64::*;

use crate::vectors::arch_simd::_128bit::i8x16;

impl PartialEq for i8x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, other.0);
            _mm_movemask_epi8(cmp) == 0xffff
        }
    }
}

impl Default for i8x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i8x16(_mm_setzero_si128()) }
    }
}

impl VecTrait<i8> for i8x16 {
    const SIZE: usize = 16;
    type Base = i8;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let mut res = [0i8; 16];
            let x: [i8; 16] = std::mem::transmute(self.0);
            let y: [i8; 16] = std::mem::transmute(a.0);
            let z: [i8; 16] = std::mem::transmute(b.0);
            for i in 0..16 {
                res[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            i8x16(_mm_loadu_si128(res.as_ptr() as *const __m128i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe {
            let sum = _mm_sad_epu8(self.0, _mm_setzero_si128());
            _mm_cvtsi128_si32(sum) as i8
        }
    }
    #[inline(always)]
    fn splat(val: i8) -> i8x16 {
        unsafe { i8x16(_mm_set1_epi8(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i8) -> Self {
        unsafe { i8x16(_mm_loadu_si128(ptr as *const __m128i)) }
    }
}

impl SimdCompare for i8x16 {
    type SimdMask = i8x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x16 {
        unsafe { i8x16(_mm_cmpeq_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x16 {
        unsafe {
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_xor_si128(eq, _mm_set1_epi8(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x16 {
        unsafe { i8x16(_mm_cmplt_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x16 {
        unsafe {
            let lt = _mm_cmplt_epi8(self.0, other.0);
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_or_si128(lt, eq))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x16 {
        unsafe { i8x16(_mm_cmpgt_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x16 {
        unsafe {
            let gt = _mm_cmpgt_epi8(self.0, other.0);
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_or_si128(gt, eq))
        }
    }
}

impl std::ops::Add for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Div for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                assert!(b[i] != 0, "division by zero");
                result[i] = a[i] / b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i8x16(_mm_sign_epi8(self.0, _mm_set1_epi8(-1))) }
    }
}
impl std::ops::BitAnd for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i8x16(_mm_xor_si128(self.0, _mm_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl SimdMath<i8> for i8x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i8x16(_mm_max_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i8x16(_mm_min_epi8(self.0, other.0)) }
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
        unsafe { i8x16(_mm_abs_epi8(self.0)) }
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
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i].pow(b[i] as u32);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(i8x16::splat(0)) + alpha * self.min(i8x16::splat(0))
    }
}

impl FloatOutBinary2 for i8x16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i8")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for i8x16");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                if b[i] < 0 {
                    panic!("Power operation is not supported for negative i8");
                }
                result[i] = a[i].pow(b[i] as u32);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl Eval2 for i8x16 {
    type Output = i8x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i8x16::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm_cmpeq_epi8(self.0, _mm_setzero_si128());
            let result = _mm_andnot_si128(eq, _mm_set1_epi8(1));
            Self(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i8x16::default()
    }
}
