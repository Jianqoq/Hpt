use std::ops::Index;

use crate::VecTrait;


/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct boolx16(pub(crate) [bool; 16]);

impl VecTrait<bool> for boolx16 {
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    fn partial_load(ptr: *const bool, num_elem: usize) -> Self {
        let mut result = [false; 16];
        for i in 0..num_elem {
            result[i] = unsafe { *ptr.add(i) };
        }
        boolx16(result)
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut bool, num_elem: usize) {
        for i in 0..num_elem {
            unsafe { *ptr.add(i) = self.0[i] };
        }
    }
}


impl std::ops::Add for boolx16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] || rhs.0[i];
        }
        ret
    }
}

impl std::ops::Mul for boolx16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] && rhs.0[i];
        }
        ret
    }
}

impl Index<usize> for boolx16 {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self as *const _ as *const bool;
        unsafe { &*ptr.add(index) }
    }
}