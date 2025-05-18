
use std::arch::aarch64::*;

use crate::{simd::_128bit::common::i64x2::i64x2, VecTrait};

impl VecTrait<i64> for i64x2 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(a.0);
            let arr3: [i64; 2] = std::mem::transmute(b.0);
            let mut arr4: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            return i64x2(vld1q_s64(arr4.as_ptr()));
        }
    }
    #[inline(always)]
    fn splat(val: i64) -> i64x2 {
        unsafe { i64x2(vdupq_n_s64(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    fn partial_load(ptr: *const i64, num_elem: usize) -> Self {
        let mut result = Self::splat(i64::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut i64, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut i64, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const i64, ptr, num_elem);
        }
    }
}

impl std::ops::Add for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vaddq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i].wrapping_mul(arr2[i]);
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
}