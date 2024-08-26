use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct bf16x16(pub(crate) [half::bf16; 16]);

impl VecTrait<half::bf16> for bf16x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::bf16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::bf16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut half::bf16 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut half::bf16 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> half::bf16 {
        self.0.iter().sum()
    }
}
impl VecSize for bf16x16 {
    const SIZE: usize = 16;
}
impl Init<half::bf16> for bf16x16 {
    fn splat(val: half::bf16) -> bf16x16 {
        bf16x16([val; 16])
    }

    unsafe fn from_ptr(ptr: *const half::bf16) -> Self {
        let mut arr = [half::bf16::default(); 16];
        arr.copy_from_slice(std::slice::from_raw_parts(ptr, 16));
        bf16x16(arr)
    }
}
impl IntoVec<bf16x16> for bf16x16 {
    fn into_vec(self) -> bf16x16 {
        self
    }
}
impl std::ops::Add for bf16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for bf16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for bf16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for bf16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for bf16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
