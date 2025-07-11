use crate::{
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOutUnary2},
};

use std::arch::x86_64::*;

use crate::simd::_256bit::i8x32;

impl PartialEq for i8x32 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi8(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i8x32 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i8x32(_mm256_setzero_si256()) }
    }
}

impl VecTrait<i8> for i8x32 {
    const SIZE: usize = 32;
    type Base = i8;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let mut res = [0i8; 32];
            let x: [i8; 32] = std::mem::transmute(self.0);
            let y: [i8; 32] = std::mem::transmute(a.0);
            let z: [i8; 32] = std::mem::transmute(b.0);
            for i in 0..32 {
                res[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            i8x32(_mm256_loadu_si256(res.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe {
            let sum = _mm256_sad_epu8(self.0, _mm256_setzero_si256());
            _mm256_cvtsi256_si32(sum) as i8
        }
    }
    #[inline(always)]
    fn splat(val: i8) -> i8x32 {
        unsafe { i8x32(_mm256_set1_epi8(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i8) -> Self {
        i8x32(_mm256_loadu_si256(ptr as *const __m256i))
    }
}

impl SimdCompare for i8x32 {
    type SimdMask = i8x32;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x32 {
        unsafe { i8x32(_mm256_cmpeq_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x32 {
        unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            i8x32(_mm256_xor_si256(eq, _mm256_set1_epi8(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x32 {
        unsafe { i8x32(_mm256_cmpgt_epi8(other.0, self.0)) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x32 {
        unsafe {
            let gt = _mm256_cmpgt_epi8(self.0, other.0);
            i8x32(_mm256_xor_si256(gt, _mm256_set1_epi8(-1)))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x32 {
        unsafe { i8x32(_mm256_cmpgt_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x32 {
        unsafe {
            let gt = _mm256_cmpgt_epi8(self.0, other.0);
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            i8x32(_mm256_or_si256(gt, eq))
        }
    }
}

impl SimdSelect<i8x32> for i8x32 {
    #[inline(always)]
    fn select(&self, true_val: i8x32, false_val: i8x32) -> i8x32 {
        unsafe {
            i8x32(_mm256_blendv_epi8(
                false_val.0,
                true_val.0,
                std::mem::transmute(self.0),
            ))
        }
    }
}

impl std::ops::Add for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                assert!(b[i] != 0, "division by zero");
                result[i] = a[i] / b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i] % b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i8x32(_mm256_sign_epi8(self.0, _mm256_set1_epi8(-1))) }
    }
}
impl std::ops::BitAnd for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i8x32(_mm256_xor_si256(self.0, _mm256_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl SimdMath<i8> for i8x32 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i8x32(_mm256_max_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i8x32(_mm256_min_epi8(self.0, other.0)) }
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
        unsafe { i8x32(_mm256_abs_epi8(self.0)) }
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
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 32];
            for i in 0..32 {
                result[i] = a[i].pow(b[i] as u32);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl FloatOutBinary2 for i8x32 {
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
        panic!("Hypot operation is not supported for i8x32");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                if b[i] < 0 {
                    panic!("Power operation is not supported for negative i8");
                }
                result[i] = a[i].pow(b[i] as u32);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl NormalOutUnary2 for i8x32 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        i8x32(unsafe { _mm256_abs_epi8(self.0) })
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
        unsafe { Self(_mm256_sub_epi8(_mm256_setzero_si256(), self.0)) }
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
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        self.abs() * rhs.signum()
    }
}

impl Eval2 for i8x32 {
    type Output = i8x32;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i8x32::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, _mm256_setzero_si256());
            let result = _mm256_andnot_si256(eq, _mm256_set1_epi8(1));
            Self(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i8x32::default()
    }
}
