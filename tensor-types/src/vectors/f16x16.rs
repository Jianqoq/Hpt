use std::simd::{cmp::SimdPartialEq, Simd};
use std::simd::cmp::SimdPartialOrd;
use crate::into_vec::IntoVec;

use super::{traits::{ Init, VecSize, VecTrait }, u16x16::u16x16};

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct f16x16(pub(crate) [half::f16; 16]);

impl VecTrait<half::f16> for f16x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::f16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut half::f16 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut half::f16 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }
    
    fn extract(self, idx: usize) -> half::f16 {
        self.0[idx]
    }
}
impl VecSize for f16x16 {
    const SIZE: usize = 16;
}
impl Init<half::f16> for f16x16 {
    fn splat(val: half::f16) -> f16x16 {
        f16x16([val; 16])
    }

    unsafe fn from_ptr(ptr: *const half::f16) -> Self {
        let mut arr = [half::f16::default(); 16];
        arr.copy_from_slice(std::slice::from_raw_parts(ptr, 16));
        f16x16(arr)
    }
}
impl IntoVec<f16x16> for f16x16 {
    fn into_vec(self) -> f16x16 {
        self
    }
}

impl f16x16 {
    pub fn is_nan(&self) -> u16x16 {
        let x = u16x16::splat(0x7C00u16);
        let y = u16x16::splat(0x03FFu16);
        let i: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
    
        let and = i & x.0;
        let eq = and.simd_eq(x.0);
    
        let and2 = i & y.0;
        let neq_zero = and2.simd_ne(u16x16::splat(0).0);
    
        let result = eq & neq_zero;
    
        unsafe { std::mem::transmute(result) }
    }
    pub fn is_infinite(&self) -> u16x16 {
        let x = u16x16::splat(0x7C00u16);
        let y = u16x16::splat(0x03FFu16);
        let i: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
    
        let and = i & x.0;
        let eq = and.simd_eq(x.0);
    
        let and2 = i & y.0;
        let eq_zero = and2.simd_eq(u16x16::splat(0).0);
    
        let result = eq & eq_zero;
    
        unsafe { std::mem::transmute(result) }
    }
    pub fn simd_eq(&self, other: Self) -> u16x16 {
        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 16> = unsafe { std::mem::transmute(other.0) };
        let eq = x.simd_eq(y);
        unsafe { std::mem::transmute(eq) }
    }
    pub fn simd_ne(&self, other: Self) -> u16x16 {
        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 16> = unsafe { std::mem::transmute(other.0) };
        let ne = x.simd_ne(y);
        unsafe { std::mem::transmute(ne) }
    }
    pub fn simd_lt(&self, other: Self) -> u16x16 {
        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 16> = unsafe { std::mem::transmute(other.0) };
        let lt = x.simd_lt(y);
        unsafe { std::mem::transmute(lt) }
    }
    pub fn simd_le(&self, other: Self) -> u16x16 {
        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 16> = unsafe { std::mem::transmute(other.0) };
        let le = x.simd_le(y);
        unsafe { std::mem::transmute(le) }
    }
    pub fn simd_gt(&self, other: Self) -> u16x16 {
        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 16> = unsafe { std::mem::transmute(other.0) };
        let gt = x.simd_gt(y);
        unsafe { std::mem::transmute(gt) }
    }
    pub fn simd_ge(&self, other: Self) -> u16x16 {
        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 16> = unsafe { std::mem::transmute(other.0) };
        let ge = x.simd_ge(y);
        unsafe { std::mem::transmute(ge) }
    }
}

impl std::ops::Add for f16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}

impl std::ops::Sub for f16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}

impl std::ops::Mul for f16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}

impl std::ops::Div for f16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for f16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}