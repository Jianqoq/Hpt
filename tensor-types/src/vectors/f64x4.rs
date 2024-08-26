use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct f64x4(wide::f64x4);

impl Deref for f64x4 {
    type Target = wide::f64x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<f64> for f64x4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        self.as_array_mut().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f64 {
        self.as_array_ref().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f64 {
        self.as_array_mut().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut f64 {
        self.as_array_ref().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        self.as_array_ref().iter().sum()
    }
}
impl VecSize for f64x4 {
    const SIZE: usize = 4;
}
impl Init<f64> for f64x4 {
    fn splat(val: f64) -> f64x4 {
        f64x4(wide::f64x4::splat(val))
    }

    unsafe fn from_ptr(ptr: *const f64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_pd(ptr as *const _)) }
    }
}
impl IntoVec<f64x4> for f64x4 {
    fn into_vec(self) -> f64x4 {
        self
    }
}