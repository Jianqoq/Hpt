use std::arch::x86_64::*;

use crate::{
    arch_simd::_256bit::i64x4::i64x4,
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
};

use super::usizex4::usizex4;

/// a vector of 4 isize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct isizex4(pub(crate) __m256i);

impl Default for isizex4 {
    fn default() -> Self {
        isizex4(unsafe { _mm256_setzero_si256() })
    }
}

impl PartialEq for isizex4 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let cmp = _mm256_cmpeq_epi64(self.0, other.0);
                _mm256_movemask_epi8(cmp) == -1
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let cmp = _mm256_cmpeq_epi32(self.0, other.0);
                _mm256_movemask_epi8(cmp) == -1
            }
        }
    }
}

impl VecTrait<isize> for isizex4 {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 4;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 4;
    type Base = isize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 4] = std::mem::transmute(self.0);
                let arr_a: [i64; 4] = std::mem::transmute(a.0);
                let arr_b: [i64; 4] = std::mem::transmute(b.0);
                let ret = [
                    arr[0] * arr_a[0] + arr_b[0],
                    arr[1] * arr_a[1] + arr_b[1],
                    arr[2] * arr_a[2] + arr_b[2],
                    arr[3] * arr_a[3] + arr_b[3],
                ];
                isizex4(std::mem::transmute(ret))
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
                isizex4(std::mem::transmute(ret))
            }
        }
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 4] = std::mem::transmute(self.0);
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
    fn splat(val: isize) -> isizex4 {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(unsafe { _mm256_set1_epi64x(val as i64) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(unsafe { _mm256_set1_epi32(val as i32) })
        }
    }
}

impl isizex4 {
    #[allow(unused)]
    #[cfg(target_pointer_width = "64")]
    fn as_array(&self) -> [isize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
    #[allow(unused)]
    #[cfg(target_pointer_width = "32")]
    fn as_array(&self) -> [isize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for isizex4 {
    type SimdMask = isizex4;
    fn simd_eq(self, other: Self) -> isizex4 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_eq(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_eq(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_ne(self, other: Self) -> isizex4 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_ne(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_ne(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_lt(self, other: Self) -> isizex4 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_lt(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_lt(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_le(self, other: Self) -> isizex4 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_le(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_le(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_gt(self, other: Self) -> isizex4 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_gt(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_gt(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn simd_ge(self, other: Self) -> isizex4 {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_ge(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.simd_ge(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
}

impl std::ops::Add for isizex4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(unsafe { _mm256_add_epi64(self.0, rhs.0) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(unsafe { _mm256_add_epi32(self.0, rhs.0) })
        }
    }
}
impl std::ops::Sub for isizex4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(unsafe { _mm256_sub_epi64(self.0, rhs.0) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(unsafe { _mm256_sub_epi32(self.0, rhs.0) })
        }
    }
}
impl std::ops::Mul for isizex4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [i64; 4] = std::mem::transmute(rhs.0);
                let ret = [arr[0] * arr_rhs[0], arr[1] * arr_rhs[1], arr[2] * arr_rhs[2], arr[3] * arr_rhs[3]];
                isizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [i32; 8] = std::mem::transmute(rhs.0);
                let ret = [
                    arr[0] * arr_rhs[0],
                    arr[1] * arr_rhs[1],
                    arr[2] * arr_rhs[2],
                    arr[3] * arr_rhs[3],
                    arr[4] * arr_rhs[4],
                    arr[5] * arr_rhs[5],
                    arr[6] * arr_rhs[6],
                    arr[7] * arr_rhs[7],
                ];
                isizex4(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Div for isizex4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [i64; 4] = std::mem::transmute(rhs.0);
                let ret = [arr[0] / arr_rhs[0], arr[1] / arr_rhs[1], arr[2] / arr_rhs[2], arr[3] / arr_rhs[3]];
                isizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [i32; 8] = std::mem::transmute(rhs.0);
                let ret = [
                    arr[0] / arr_rhs[0],
                    arr[1] / arr_rhs[1],
                    arr[2] / arr_rhs[2],
                    arr[3] / arr_rhs[3],
                    arr[4] / arr_rhs[4],
                    arr[5] / arr_rhs[5],
                    arr[6] / arr_rhs[6],
                    arr[7] / arr_rhs[7],
                ];
                isizex4(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Rem for isizex4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [i64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [i64; 4] = std::mem::transmute(rhs.0);
                let ret = [arr[0] % arr_rhs[0], arr[1] % arr_rhs[1], arr[2] % arr_rhs[2], arr[3] % arr_rhs[3]];
                isizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [i32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [i32; 8] = std::mem::transmute(rhs.0);
                let ret = [
                    arr[0] % arr_rhs[0],
                    arr[1] % arr_rhs[1],
                    arr[2] % arr_rhs[2],
                    arr[3] % arr_rhs[3],
                    arr[4] % arr_rhs[4],
                    arr[5] % arr_rhs[5],
                    arr[6] % arr_rhs[6],
                    arr[7] % arr_rhs[7],
                ];
                isizex4(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Neg for isizex4 {
    type Output = Self;
    fn neg(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(unsafe { _mm256_sub_epi64(_mm256_setzero_si256(), self.0) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), self.0) })
        }
    }
}
impl std::ops::BitAnd for isizex4 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { isizex4(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for isizex4 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { isizex4(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for isizex4 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { isizex4(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for isizex4 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { isizex4(_mm256_xor_si256(self.0, _mm256_set1_epi64x(-1))) }
    }
}
impl std::ops::Shl for isizex4 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let a: [i64; 4] = std::mem::transmute(self.0);
                let b: [i64; 4] = std::mem::transmute(rhs.0);
                let mut result = [0; 4];
                for i in 0..4 {
                    result[i] = a[i] << b[i];
                }
                isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let a: [i32; 8] = std::mem::transmute(self.0);
                let b: [i32; 8] = std::mem::transmute(rhs.0);
                let mut result = [0; 8];
                for i in 0..8 {
                    result[i] = a[i] << b[i];
                }
                isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
            }
        }
    }
}
impl std::ops::Shr for isizex4 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let a: [i64; 4] = std::mem::transmute(self.0);
                let b: [i64; 4] = std::mem::transmute(rhs.0);
                let mut result = [0; 4];
                for i in 0..4 {
                    result[i] = a[i] >> b[i];
                }
                isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let a: [i32; 8] = std::mem::transmute(self.0);
                let b: [i32; 8] = std::mem::transmute(rhs.0);
                let mut result = [0; 8];
                for i in 0..8 {
                    result[i] = a[i] >> b[i];
                }
                isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
            }
        }
    }
}
impl SimdMath<isize> for isizex4 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                isizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn relu(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x4 = std::mem::transmute(self.0);
                isizex4(std::mem::transmute(lhs.relu()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                isizex4(std::mem::transmute(lhs.relu()))
            }
        }
    }
    fn relu6(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x4 = std::mem::transmute(self.0);
                isizex4(std::mem::transmute(lhs.relu6()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                isizex4(std::mem::transmute(lhs.relu6()))
            }
        }
    }
}

impl VecConvertor for isizex4 {
    fn to_isize(self) -> isizex4 {
        self
    }
    fn to_usize(self) -> usizex4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_i64(self) -> i64x4 {
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
    fn to_f64(self) -> super::f64x4::f64x4 {
        self.to_i64().to_f64()
    }
}
