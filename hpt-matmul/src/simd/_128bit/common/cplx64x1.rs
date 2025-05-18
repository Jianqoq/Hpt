
use num_complex::Complex64;

use crate::VecTrait;

/// a vector of 1 Complex64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct cplx64x1(pub(crate) [Complex64; 1]);

impl VecTrait<Complex64> for cplx64x1 {
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self
    }
    #[inline(always)]
    fn splat(val: Complex64) -> cplx64x1 {
        cplx64x1([val; 1])
    }
    #[inline(always)]
    #[cfg(target_feature = "neon")]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a.0[LANE as usize]);
        self.mul_add(val, b)
    }
    
    fn partial_load(ptr: *const Complex64, num_elem: usize) -> Self {
        let mut result = [Complex64::ZERO; 1];
        for i in 0..num_elem {
            result[i] = unsafe { *ptr.add(i) };
        }
        cplx64x1(result)
    }
    
    fn partial_store(self, ptr: *mut Complex64, num_elem: usize) {
        for i in 0..num_elem {
            unsafe { *ptr.add(i) = self.0[i] };
        }
    }
    
}

impl std::ops::Add for cplx64x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x1 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}