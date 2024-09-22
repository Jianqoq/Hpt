use std::ops::{Index, IndexMut};

use num_complex::Complex32;

use crate::vectors::traits::{ Init, VecCommon, VecTrait };

/// a vector of 2 Complex32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct cplx32x2(pub(crate) [Complex32; 2]);

impl VecTrait<Complex32> for cplx32x2 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex32]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex32 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex32 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut Complex32 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> Complex32 {
        self.0.iter().sum()
    }
    
    fn extract(self, idx: usize) -> Complex32 {
        self.0[idx]
    }
}
impl VecCommon for cplx32x2 {
    const SIZE: usize = 2;
    
    type Base = Complex32;
}
impl Init<Complex32> for cplx32x2 {
    fn splat(val: Complex32) -> cplx32x2 {
        cplx32x2([val; 2])
    }
}
impl Index<usize> for cplx32x2 {
    type Output = Complex32;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}
impl IndexMut<usize> for cplx32x2 {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.0[idx]
    }
}
impl std::ops::Add for cplx32x2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx32x2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx32x2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx32x2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for cplx32x2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}
impl std::ops::Rem for cplx32x2 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}