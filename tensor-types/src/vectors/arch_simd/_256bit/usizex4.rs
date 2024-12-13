use crate::{
    arch_simd::_256bit::u64x4::u64x4, convertion::VecConvertor, traits::{SimdCompare, SimdMath, VecTrait}
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{i64x4::i64x4, isizex4::isizex4};

/// a vector of 4 usize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct usizex4(pub(crate) __m256i);

impl PartialEq for usizex4 {
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

impl Default for usizex4 {
    fn default() -> Self {
        usizex4(unsafe { _mm256_setzero_si256() })
    }
}

impl VecTrait<usize> for usizex4 {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 4;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 8;
    type Base = usize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                let arr_a: [u64; 4] = std::mem::transmute(a.0);
                let arr_b: [u64; 4] = std::mem::transmute(b.0);
                let ret = [arr[0] * arr_a[0] + arr_b[0], arr[1] * arr_a[1] + arr_b[1], arr[2] * arr_a[2] + arr_b[2], arr[3] * arr_a[3] + arr_b[3]];
                usizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                let arr_a: [u32; 8] = std::mem::transmute(a.0);
                let arr_b: [u32; 8] = std::mem::transmute(b.0);
                let ret = [
                    arr[0] * arr_a[0] + arr_b[0],
                    arr[1] * arr_a[1] + arr_b[1],
                    arr[2] * arr_a[2] + arr_b[2],
                    arr[3] * arr_a[3] + arr_b[3],
                    arr[4] * arr_a[4] + arr_b[4],
                    arr[5] * arr_a[5] + arr_b[5],
                    arr[6] * arr_a[6] + arr_b[6],
                    arr[7] * arr_a[7] + arr_b[7],
                ];
                usizex4(std::mem::transmute(ret))
            }
        }
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                arr.iter().sum::<u64>() as usize
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                arr.iter().sum::<u32>() as usize
            }
        }
    }
    fn splat(val: usize) -> usizex4 {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe { usizex4(_mm256_set1_epi64x(val as i64)) }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe { usizex4(_mm256_set1_epi32(val as i32)) }
        }
    }

}

impl usizex4 {
    #[allow(unused)]
    fn as_array(&self) -> [usize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for usizex4 {
    type SimdMask = isizex4;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_eq(rhs))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x8 = std::mem::transmute(self.0);
                let rhs: i32x8 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_eq(rhs))
            }
        }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_ne(rhs))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x8 = std::mem::transmute(self.0);
                let rhs: i32x8 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_ne(rhs))
            }
        }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_lt(rhs))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x8 = std::mem::transmute(self.0);
                let rhs: i32x8 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_lt(rhs))
            }
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_le(rhs))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x8 = std::mem::transmute(self.0);
                let rhs: i32x8 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_le(rhs))
            }
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_gt(rhs))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x8 = std::mem::transmute(self.0);
                let rhs: i32x8 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_gt(rhs))
            }
        }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            #[cfg(target_pointer_width = "64")]
            {
                let lhs: i64x4 = std::mem::transmute(self.0);
                let rhs: i64x4 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_ge(rhs))
            }
            #[cfg(target_pointer_width = "32")]
            {
                let lhs: i32x8 = std::mem::transmute(self.0);
                let rhs: i32x8 = std::mem::transmute(other.0);
                std::mem::transmute(lhs.simd_ge(rhs))
            }
        }
    }
}

impl std::ops::Add for usizex4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe { usizex4(_mm256_add_epi64(self.0, rhs.0)) }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe { usizex4(_mm256_add_epi32(self.0, rhs.0)) }
        }
    }
}
impl std::ops::Sub for usizex4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe { usizex4(_mm256_sub_epi64(self.0, rhs.0)) }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe { usizex4(_mm256_sub_epi32(self.0, rhs.0)) }
        }
    }
}
impl std::ops::Mul for usizex4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [u64; 4] = std::mem::transmute(rhs.0);
                let ret = [arr[0] * arr_rhs[0], arr[1] * arr_rhs[1], arr[2] * arr_rhs[2], arr[3] * arr_rhs[3]];
                usizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [u32; 8] = std::mem::transmute(rhs.0);
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
                usizex4(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Div for usizex4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [u64; 4] = std::mem::transmute(rhs.0);
                let ret = [arr[0] / arr_rhs[0], arr[1] / arr_rhs[1], arr[2] / arr_rhs[2], arr[3] / arr_rhs[3]];
                usizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [u32; 8] = std::mem::transmute(rhs.0);
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
                usizex4(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::Rem for usizex4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [u64; 4] = std::mem::transmute(rhs.0);
                let ret = [arr[0] % arr_rhs[0], arr[1] % arr_rhs[1], arr[2] % arr_rhs[2], arr[3] % arr_rhs[3]];
                usizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [u32; 8] = std::mem::transmute(rhs.0);
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
                usizex4(std::mem::transmute(ret))
            }
        }
    }
}
impl std::ops::BitAnd for usizex4 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        unsafe { usizex4(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for usizex4 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        unsafe { usizex4(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for usizex4 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { usizex4(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for usizex4 {
    type Output = Self;
    fn not(self) -> Self {
        unsafe { usizex4(_mm256_xor_si256(self.0, _mm256_set1_epi64x(-1))) }
    }
}
impl std::ops::Shl for usizex4 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [u64; 4] = std::mem::transmute(rhs.0);
                let mut result = [0; 4];
                for i in 0..4 {
                    result[i] = arr[i] << arr_rhs[i];
                }
                usizex4(std::mem::transmute(result))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [u32; 8] = std::mem::transmute(rhs.0);
                let mut result = [0; 8];
                for i in 0..8 {
                    result[i] = arr[i] << arr_rhs[i];
                }
                usizex4(std::mem::transmute(result))
            }
        }
    }
}
impl std::ops::Shr for usizex4 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let arr: [u64; 4] = std::mem::transmute(self.0);
                let arr_rhs: [u64; 4] = std::mem::transmute(rhs.0);
                let mut result = [0; 4];
                for i in 0..4 {
                    result[i] = arr[i] >> arr_rhs[i];
                }
                usizex4(std::mem::transmute(result))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let arr: [u32; 8] = std::mem::transmute(self.0);
                let arr_rhs: [u32; 8] = std::mem::transmute(rhs.0);
                let mut result = [0; 8];
                for i in 0..8 {
                    result[i] = arr[i] >> arr_rhs[i];
                }
                usizex4(std::mem::transmute(result))
            }
        }
    }
}
impl SimdMath<usize> for usizex4 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                let rhs: u64x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                usizex4(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                let rhs: u32x8 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                usizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                let rhs: u64x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                usizex4(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                let rhs: u32x8 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                usizex4(std::mem::transmute(ret.0))
            }
        }
    }
    fn relu(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu()))
            }
        }
    }
    fn relu6(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu6()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu6()))
            }
        }
    }
}

impl VecConvertor for usizex4 {
    fn to_usize(self) -> usizex4 {
        self
    }
    #[cfg(target_pointer_width = "64")]
    fn to_u64(self) -> u64x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_u32(self) -> u32x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_f32(self) -> super::f32x8::f32x8 {
        self.to_u32().to_f32()
    }
    #[cfg(target_pointer_width = "64")]
    fn to_i64(self) -> i64x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_i32(self) -> i32x8 {
        unsafe { std::mem::transmute(self) }
    }
}