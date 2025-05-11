use std::ops::Index;


/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct boolx16(pub(crate) [bool; 16]);

impl boolx16 {
    #[inline(always)]
    pub(crate) fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    pub(crate) fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const bool) -> Self {
        let mut result = [false; 16];
        for i in 0..16 {
            result[i] = unsafe { *ptr.add(i) };
        }
        boolx16(result)
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