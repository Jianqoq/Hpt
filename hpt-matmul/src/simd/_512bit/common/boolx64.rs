use crate::VecTrait;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct boolx64(pub(crate) [bool; 64]);

impl VecTrait<bool> for boolx64 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut ret = Self([false; 64]);
        for i in 0..64 {
            ret.0[i] = (self.0[i] && a.0[i]) || b.0[i];
        }
        ret
    }
    #[inline(always)]
    fn splat(val: bool) -> boolx64 {
        boolx64([val; 64])
    }

    fn partial_load(ptr: *const bool, num_elem: usize) -> Self {
        let mut ret = Self([false; 64]);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, ret.0.as_mut_ptr(), num_elem);
        }
        ret
    }

    fn partial_store(self, ptr: *mut bool, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, num_elem);
        }
    }
}

impl std::ops::Add for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = Self([false; 64]);
        for i in 0..64 {
            ret.0[i] = self.0[i] || rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = Self([false; 64]);
        for i in 0..64 {
            ret.0[i] = self.0[i] && rhs.0[i];
        }
        ret
    }
}
