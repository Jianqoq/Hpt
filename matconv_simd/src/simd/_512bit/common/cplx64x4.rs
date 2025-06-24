use num_complex::Complex64;

use crate::VecTrait;

/// a vector of 2 cplx64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct cplx64x4(pub(crate) [Complex64; 4]);

impl VecTrait<Complex64> for cplx64x4 {
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self.0[2] = self.0[2] * a.0[2] + b.0[2];
        self.0[3] = self.0[3] * a.0[3] + b.0[3];
        self
    }
    #[inline(always)]
    fn splat(val: Complex64) -> cplx64x4 {
        cplx64x4([val; 4])
    }

    fn partial_load(ptr: *const Complex64, num_elem: usize) -> Self {
        let mut ret = cplx64x4::default();
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, ret.0.as_mut_ptr(), num_elem);
        }
        ret
    }

    fn partial_store(self, ptr: *mut Complex64, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, num_elem);
        }
    }
}

impl std::ops::Add for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
