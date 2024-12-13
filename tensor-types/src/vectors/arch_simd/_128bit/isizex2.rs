use std::arch::x86_64::*;

use crate::{
    arch_simd::_128bit::i64x2::i64x2,
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
};

use super::usizex2::usizex2;

/// a vector of 2 isize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct isizex2(pub(crate) __m128i);

impl Default for isizex2 {
    fn default() -> Self {
        isizex2(unsafe { _mm_setzero_si128() })
    }
}

impl PartialEq for isizex2 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let cmp = _mm_cmpeq_epi64(self.0, other.0);
                _mm_movemask_epi8(cmp) == -1
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let cmp = _mm_cmpeq_epi32(self.0, other.0);
                _mm_movemask_epi8(cmp) == -1
            }
        }
    }
}

impl VecTrait<isize> for isizex2 {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 2;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 4;
    type Base = isize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.0 = unsafe { _mm_loadu_si128(slice.as_ptr() as *const __m128i) };
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 2] = std::mem::transmute(self.0);
                let arr_a: [i64; 2] = std::mem::transmute(a.0);
                let arr_b: [i64; 2] = std::mem::transmute(b.0);
                let ret = [arr[0] * arr_a[0] + arr_b[0], arr[1] * arr_a[1] + arr_b[1]];
                isizex2(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 4] = std::mem::transmute(self.0);
                let arr_a: [i32; 4] = std::mem::transmute(a.0);
                let arr_b: [i32; 4] = std::mem::transmute(b.0);
                let ret = [
                    arr[0] * arr_a[0] + arr_b[0],
                    arr[1] * arr_a[1] + arr_b[1],
                    arr[2] * arr_a[2] + arr_b[2],
                    arr[3] * arr_a[3] + arr_b[3],
                ];
                isizex2(std::mem::transmute(ret))
            }
        }
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 2] = std::mem::transmute(self.0);
                arr.iter().sum::<i64>() as isize
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 4] = std::mem::transmute(self.0);
                arr.iter().sum::<i32>() as isize
            }
        }
    }
    fn splat(val: isize) -> isizex2 {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(unsafe { _mm_set1_epi64x(val as i64) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex2(unsafe { _mm_set1_epi32(val as i32) })
        }
    }
}

impl isizex2 {
    #[allow(unused)]
    #[cfg(target_pointer_width = "64")]
    fn as_array(&self) -> [isize; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
    #[allow(unused)]
    #[cfg(target_pointer_width = "32")]
    fn as_array(&self) -> [isize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for isizex2 {
    type SimdMask = isizex2;
    fn simd_eq(self, other: Self) -> isizex2 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.simd_eq(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_eq(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_ne(self, other: Self) -> isizex2 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.simd_ne(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_ne(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_lt(self, other: Self) -> isizex2 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.simd_lt(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_lt(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_le(self, other: Self) -> isizex2 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.simd_le(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_le(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_gt(self, other: Self) -> isizex2 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.simd_gt(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_gt(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_ge(self, other: Self) -> isizex2 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.simd_ge(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_ge(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
}

impl std::ops::Add for isizex2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(unsafe { _mm_add_epi64(self.0, rhs.0) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex2(unsafe { _mm_add_epi32(self.0, rhs.0) })
        }
    }
}
impl std::ops::Sub for isizex2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(unsafe { _mm_sub_epi64(self.0, rhs.0) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex2(unsafe { _mm_sub_epi32(self.0, rhs.0) })
        }
    }
}
impl std::ops::Mul for isizex2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 2] = std::mem::transmute(self.0);
                let arr_rhs: [i64; 2] = std::mem::transmute(rhs.0);
                let ret = [arr[0] * arr_rhs[0], arr[1] * arr_rhs[1]];
                isizex2(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 4] = std::mem::transmute(self.0);
                let arr_rhs: [i32; 4] = std::mem::transmute(rhs.0);
                let ret = [
                    arr[0] * arr_rhs[0],
                    arr[1] * arr_rhs[1],
                    arr[2] * arr_rhs[2],
                    arr[3] * arr_rhs[3],
                ];
                isizex2(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Div for isizex2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 2] = std::mem::transmute(self.0);
                let arr_rhs: [i64; 2] = std::mem::transmute(rhs.0);
                let ret = [arr[0] / arr_rhs[0], arr[1] / arr_rhs[1]];
                isizex2(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 4] = std::mem::transmute(self.0);
                let arr_rhs: [i32; 4] = std::mem::transmute(rhs.0);
                let ret = [
                    arr[0] / arr_rhs[0],
                    arr[1] / arr_rhs[1],
                    arr[2] / arr_rhs[2],
                    arr[3] / arr_rhs[3],
                ];
                isizex2(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Rem for isizex2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 2] = std::mem::transmute(self.0);
                let arr_rhs: [i64; 2] = std::mem::transmute(rhs.0);
                let ret = [arr[0] % arr_rhs[0], arr[1] % arr_rhs[1]];
                isizex2(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 4] = std::mem::transmute(self.0);
                let arr_rhs: [i32; 4] = std::mem::transmute(rhs.0);
                let ret = [
                    arr[0] % arr_rhs[0],
                    arr[1] % arr_rhs[1],
                    arr[2] % arr_rhs[2],
                    arr[3] % arr_rhs[3],
                ];
                isizex2(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Neg for isizex2 {
    type Output = Self;
    fn neg(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(unsafe { _mm_sub_epi64(_mm_setzero_si128(), self.0) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex2(unsafe { _mm_sub_epi32(_mm_setzero_si128(), self.0) })
        }
    }
}
impl std::ops::BitAnd for isizex2 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { isizex2(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for isizex2 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { isizex2(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for isizex2 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { isizex2(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for isizex2 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { isizex2(_mm_xor_si128(self.0, _mm_set1_epi64x(-1))) }
    }
}
impl std::ops::Shl for isizex2 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let a: [i64; 2] = std::mem::transmute(self.0);
                let b: [i64; 2] = std::mem::transmute(rhs.0);
                let mut result = [0; 2];
                for i in 0..2 {
                    result[i] = a[i] << b[i];
                }
                isizex2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let a: [i32; 4] = std::mem::transmute(self.0);
                let b: [i32; 4] = std::mem::transmute(rhs.0);
                let mut result = [0; 4];
                for i in 0..4 {
                    result[i] = a[i] << b[i];
                }
                isizex2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
            }
        }
    }
}
impl std::ops::Shr for isizex2 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let a: [i64; 2] = std::mem::transmute(self.0);
                let b: [i64; 2] = std::mem::transmute(rhs.0);
                let mut result = [0; 2];
                for i in 0..2 {
                    result[i] = a[i] >> b[i];
                }
                isizex2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let a: [i32; 4] = std::mem::transmute(self.0);
                let b: [i32; 4] = std::mem::transmute(rhs.0);
                let mut result = [0; 4];
                for i in 0..4 {
                    result[i] = a[i] >> b[i];
                }
                isizex2(_mm_loadu_si128(result.as_ptr() as *const __m128i))
            }
        }
    }
}
impl SimdMath<isize> for isizex2 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                isizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn relu(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu()))
            }
        }
    }
    fn relu6(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu6()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu6()))
            }
        }
    }
}

impl VecConvertor for isizex2 {
    fn to_isize(self) -> isizex2 {
        self
    }
    fn to_usize(self) -> usizex2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_i64(self) -> i64x2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_i32(self) -> i32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_f32(self) -> super::f32x4::f32x4 {
        self.to_i32().to_f32()
    }
    #[cfg(target_pointer_width = "64")]
    fn to_f64(self) -> super::f64x2::f64x2 {
        self.to_i64().to_f64()
    }
}
