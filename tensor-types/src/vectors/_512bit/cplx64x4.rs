use num_complex::Complex64;

use crate::into_vec::IntoVec;

use crate::vectors::traits::{ Init, VecCommon, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct cplx64x4(pub(crate) [Complex64; 4]);

impl VecTrait<Complex64> for cplx64x4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex64]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex64 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut Complex64 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> Complex64 {
        self.0.iter().sum()
    }
    
    fn extract(self, idx: usize) -> Complex64 {
        self.0[idx]
    }
}
impl VecCommon for cplx64x4 {
    const SIZE: usize = 4;
    
    type Base = Complex64;
}
impl Init<Complex64> for cplx64x4 {
    fn splat(val: Complex64) -> cplx64x4 {
        cplx64x4([val; 4])
    }
}
impl std::ops::Add for cplx64x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx64x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx64x4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}