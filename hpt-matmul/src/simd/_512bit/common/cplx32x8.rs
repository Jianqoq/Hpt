use num_complex::Complex32;

use crate::VecTrait;

/// a vector of 4 cplx32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct cplx32x8(pub(crate) [Complex32; 8]);

impl VecTrait<Complex32> for cplx32x8 {
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self.0[2] = self.0[2] * a.0[2] + b.0[2];
        self.0[3] = self.0[3] * a.0[3] + b.0[3];
        self.0[4] = self.0[4] * a.0[4] + b.0[4];
        self.0[5] = self.0[5] * a.0[5] + b.0[5];
        self.0[6] = self.0[6] * a.0[6] + b.0[6];
        self.0[7] = self.0[7] * a.0[7] + b.0[7];
        self
    }
    #[inline(always)]
    fn splat(val: Complex32) -> cplx32x8 {
        cplx32x8([val; 8])
    }
    
    fn partial_load(ptr: *const Complex32, num_elem: usize) -> Self {
        let mut ret = cplx32x8::default();
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, ret.0.as_mut_ptr(), num_elem);
        }
        ret
    }
    
    fn partial_store(self, ptr: *mut Complex32, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, num_elem);
        }
    }
}

impl std::ops::Add for cplx32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}