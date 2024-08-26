use std::ops::Rem;

use half::{ bf16, f16 };
use num_complex::{ Complex32, Complex64 };
use statrs::function::erf::erf;
use wide::*;

use crate::type_promote::{ Cmp, Eval, FloatOut, NormalOut };

pub trait VecTrait<T> {
    fn _mul_add(self, a: Self, b: Self) -> Self;
    fn copy_from_slice(&mut self, slice: &[T]);
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_mut_ptr_uncheck(&self) -> *mut T;
    fn sum(&self) -> T;
}
pub trait Init<T> {
    fn splat(val: T) -> Self;
    unsafe fn from_ptr(ptr: *const T) -> Self;
}
pub trait VecSize {
    const SIZE: usize;
}

macro_rules! impl_vectors {
    ($T:ident, $base:ident, $size:expr) => {
        impl VecTrait<$base> for $T {
            #[inline(always)]
            fn copy_from_slice(&mut self, slice: &[$base]) {
                self.as_array_mut().copy_from_slice(slice);
            }
            #[inline(always)]
            fn as_ptr(&self) -> *const $base {
                self.as_array_ref().as_ptr()
            }
            #[inline(always)]
            fn _mul_add(self, _: Self, _: Self) -> Self {
                todo!()
            }
            #[inline(always)]
            fn as_mut_ptr(&mut self) -> *mut $base {
                self.as_array_mut().as_mut_ptr()
            }
            #[inline(always)]
            fn as_mut_ptr_uncheck(&self) -> *mut $base {
                self.as_array_ref().as_ptr() as *mut _
            }
            #[inline(always)]
            fn sum(&self) -> $base {
                self.as_array_ref().iter().sum()
            }
        }
        impl VecSize for $T {
            const SIZE: usize = $size;
        }
    };
}

impl_vectors!(i8x16, i8, 16);
impl_vectors!(i8x32, i8, 32);
impl Init<i8> for i8x32 {
    fn splat(val: i8) -> i8x32 {
        i8x32::splat(val)
    }

    unsafe fn from_ptr(ptr: *const i8) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl_vectors!(u8x16, u8, 16);
impl_vectors!(i32x4, i32, 4);
impl_vectors!(i32x8, i32, 8);
impl Init<i32> for i32x8 {
    fn splat(val: i32) -> i32x8 {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_set1_epi32(val)) }
    }

    unsafe fn from_ptr(ptr: *const i32) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl_vectors!(u32x8, u32, 32);
impl Init<u32> for u32x8 {
    fn splat(val: u32) -> u32x8 {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_set1_epi32(val as i32)) }
    }

    unsafe fn from_ptr(ptr: *const u32) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl_vectors!(f32x4, f32, 4);
impl Init<f32> for f32x8 {
    #[inline(always)]
    fn splat(val: f32) -> f32x8 {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_set1_ps(val)) }
    }
#[inline(always)]
    unsafe fn from_ptr(ptr: *const f32) -> Self {
        let ret = unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_ps(ptr)) };
        ret
    }
}
impl_vectors!(f32x8, f32, 8);
impl_vectors!(f64x2, f64, 2);
impl_vectors!(f64x4, f64, 4);
impl Init<f64> for f64x4 {
    fn splat(val: f64) -> f64x4 {
        f64x4::splat(val)
    }

    unsafe fn from_ptr(ptr: *const f64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_pd(ptr as *const _)) }
    }
}
impl_vectors!(i64x2, i64, 2);
impl Init<i64> for i64x2 {
    fn splat(val: i64) -> i64x2 {
        i64x2::splat(val)
    }
    unsafe fn from_ptr(ptr: *const i64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm_load_si128(ptr as *const _)) }
    }
}
impl_vectors!(i64x4, i64, 4);
impl Init<i64> for i64x4 {
    fn splat(val: i64) -> i64x4 {
        i64x4::splat(val)
    }
    unsafe fn from_ptr(ptr: *const i64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl_vectors!(u64x2, u64, 2);
impl Init<u64> for u64x2 {
    fn splat(val: u64) -> u64x2 {
        u64x2::splat(val)
    }
    unsafe fn from_ptr(ptr: *const u64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm_load_si128(ptr as *const _)) }
    }
}
impl_vectors!(u64x4, u64, 4);
impl_vectors!(u32x4, u32, 4);
impl_vectors!(i16x8, i16, 8);
impl Init<i16> for i16x8 {
    fn splat(val: i16) -> i16x8 {
        i16x8::splat(val)
    }
    unsafe fn from_ptr(ptr: *const i16) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm_load_si128(ptr as *const _)) }
    }
}
impl_vectors!(i16x16, i16, 16);
impl Init<i16> for i16x16 {
    fn splat(val: i16) -> i16x16 {
        i16x16::splat(val)
    }
    unsafe fn from_ptr(ptr: *const i16) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl_vectors!(u16x8, u16, 8);
impl Init<u16> for u16x8 {
    fn splat(val: u16) -> u16x8 {
        u16x8::splat(val)
    }
    unsafe fn from_ptr(ptr: *const u16) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm_load_si128(ptr as *const _)) }
    }
}
impl_vectors!(u16x16, u16, 16);
impl Init<bool> for [bool; 32] {
    fn splat(val: bool) -> [bool; 32] {
        [val; 32]
    }
    unsafe fn from_ptr(ptr: *const bool) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 32).try_into().unwrap() }
    }
}
impl VecTrait<bool> for [bool; 32] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const bool {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut bool {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        unimplemented!()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut bool {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [bool; 32] {
    const SIZE: usize = 32;
}
impl Init<u8> for [u8; 32] {
    fn splat(val: u8) -> [u8; 32] {
        [val; 32]
    }
    unsafe fn from_ptr(ptr: *const u8) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 32).try_into().unwrap() }
    }
}
impl VecTrait<u8> for [u8; 32] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut u8 {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [u8; 32] {
    const SIZE: usize = 32;
}

impl Init<isize> for [isize; 8] {
    fn splat(val: isize) -> [isize; 8] {
        [val; 8]
    }
    unsafe fn from_ptr(ptr: *const isize) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 8).try_into().unwrap() }
    }
}
impl VecTrait<isize> for [isize; 8] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const isize {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut isize {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut isize {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [isize; 8] {
    const SIZE: usize = 8;
}

impl Init<usize> for [usize; 8] {
    fn splat(val: usize) -> [usize; 8] {
        [val; 8]
    }
    unsafe fn from_ptr(ptr: *const usize) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 8).try_into().unwrap() }
    }
}
impl VecTrait<usize> for [usize; 8] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const usize {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut usize {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut usize {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [usize; 8] {
    const SIZE: usize = 8;
}

impl Init<f16> for [f16; 32] {
    fn splat(val: f16) -> [f16; 32] {
        [val; 32]
    }
    unsafe fn from_ptr(ptr: *const f16) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 32).try_into().unwrap() }
    }
}
impl VecTrait<f16> for [f16; 32] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f16]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f16 {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f16 {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> f16 {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut f16 {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [f16; 32] {
    const SIZE: usize = 32;
}

impl Init<bf16> for [bf16; 32] {
    fn splat(val: bf16) -> [bf16; 32] {
        [val; 32]
    }
    unsafe fn from_ptr(ptr: *const bf16) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 32).try_into().unwrap() }
    }
}
impl VecTrait<bf16> for [bf16; 32] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bf16]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const bf16 {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut bf16 {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> bf16 {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut bf16 {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [bf16; 32] {
    const SIZE: usize = 32;
}

impl Init<Complex32> for [Complex32; 4] {
    fn splat(val: Complex32) -> [Complex32; 4] {
        [val; 4]
    }
    unsafe fn from_ptr(ptr: *const Complex32) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 4).try_into().unwrap() }
    }
}
impl VecTrait<Complex32> for [Complex32; 4] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex32]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex32 {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex32 {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> Complex32 {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut Complex32 {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [Complex32; 4] {
    const SIZE: usize = 4;
}

impl Init<Complex64> for [Complex64; 2] {
    fn splat(val: Complex64) -> [Complex64; 2] {
        [val; 2]
    }
    unsafe fn from_ptr(ptr: *const Complex64) -> Self {
        unsafe { std::slice::from_raw_parts(ptr, 2).try_into().unwrap() }
    }
}
impl VecTrait<Complex64> for [Complex64; 2] {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex64]) {
        self[..slice.len()].copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex64 {
        self[..].as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self[..].as_mut_ptr()
    }
    #[inline(always)]
    fn sum(&self) -> Complex64 {
        self.iter().sum()
    }
    
    fn as_mut_ptr_uncheck(&self) -> *mut Complex64 {
        self[..].as_ptr() as *mut _
    }
}
impl VecSize for [Complex64; 2] {
    const SIZE: usize = 2;
}

impl FloatOut for f32x8 {
    type Output = f32x8;

    fn _div(self, rhs: Self) -> Self::Output {
        self / rhs
    }

    fn _exp(self) -> Self::Output {
        self.exp()
    }

    fn _exp2(self) -> Self::Output {
        todo!()
    }

    fn _ln(self) -> Self::Output {
        self.ln()
    }

    fn _log(self, _: Self) -> Self::Output {
        todo!()
    }

    fn _celu(self, _: Self::Output) -> Self::Output {
        todo!()
    }

    fn _log2(self) -> Self::Output {
        self.log2()
    }

    fn _log10(self) -> Self::Output {
        self.log10()
    }

    fn _sqrt(self) -> Self::Output {
        self.sqrt()
    }

    fn _sin(self) -> Self::Output {
        self.sin()
    }

    fn _cos(self) -> Self::Output {
        self.cos()
    }

    fn _tan(self) -> Self::Output {
        self.tan()
    }

    fn _asin(self) -> Self::Output {
        self.asin()
    }

    fn _acos(self) -> Self::Output {
        self.acos()
    }

    fn _atan(self) -> Self::Output {
        self.atan()
    }

    fn _sinh(self) -> Self::Output {
        self.exp() - (-self).exp() / 2.0
    }

    fn _cosh(self) -> Self::Output {
        self.exp() + (-self).exp() / 2.0
    }

    fn _tanh(self) -> Self::Output {
        self._sinh() / self._cosh()
    }

    fn _asinh(self) -> Self::Output {
        (self + (self * self + 1.0).sqrt()).ln()
    }

    fn _acosh(self) -> Self::Output {
        (self + (self * self - 1.0).sqrt()).ln()
    }

    fn _atanh(self) -> Self::Output {
        (1.0 + self) / (1.0 - self).ln() / 2.0
    }

    fn _recip(self) -> Self::Output {
        self.recip()
    }

    fn _erf(self) -> Self::Output {
        let mut res = self.clone();
        res.as_array_mut()
            .iter_mut()
            .for_each(|x| {
                *x = erf(*x as f64) as f32;
            });
        res
    }

    fn _sigmoid(self) -> Self::Output {
        todo!()
    }

    fn _elu(self, _: Self::Output) -> Self::Output {
        todo!()
    }

    fn _leaky_relu(self, _: Self::Output) -> Self::Output {
        todo!()
    }

    fn _relu(self) -> Self::Output {
        self.max(Self::Output::splat(0.0))
    }

    fn _gelu(self) -> Self::Output {
        let sqrt2_over_2 = f32x8::splat(std::f32::consts::FRAC_1_SQRT_2);
        0.5 * self * (f32x8::splat(1.0) + (self * sqrt2_over_2)._erf())
    }

    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
        let mask = self.cmp_gt(Self::Output::splat(0.0));
        (scale * self).blend(scale * alpha * (self.exp() - 1.0), mask)
    }

    fn _hard_sigmoid(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }

    fn _relu6(self) -> Self::Output {
        todo!()
    }

    fn _hard_swish(self) -> Self::Output {
        todo!()
    }

    fn _softplus(self) -> Self::Output {
        todo!()
    }

    fn _softsign(self) -> Self::Output {
        todo!()
    }

    fn _mish(self) -> Self::Output {
        todo!()
    }

    fn _cbrt(self) -> Self::Output {
        todo!()
    }
}
impl NormalOut for f32x8 {
    type Output = f32x8;
    fn _add(self, rhs: Self) -> Self::Output {
        self + rhs
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
    fn _pow(self, rhs: Self) -> Self::Output {
        self.pow_f32x8(rhs)
    }
    fn _rem(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _square(self) -> Self::Output {
        self * self
    }
    fn _abs(self) -> Self {
        self.abs()
    }
    fn _ceil(self) -> Self::Output {
        todo!()
    }
    fn _floor(self) -> Self::Output {
        todo!()
    }
    fn _sign(self) -> Self::Output {
        self.sign_bit()
    }
    fn _max(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
    fn _min(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self.round()
    }
}
impl Eval for f32x8 {
    type Output = bool;

    fn _is_nan(&self) -> Self::Output {
        self.is_nan().all()
    }

    fn _is_true(&self) -> Self::Output {
        todo!()
    }

    fn _is_inf(&self) -> Self::Output {
        self.is_inf().all()
    }
}
impl Cmp for f32x8 {
    fn _eq(self, rhs: Self) -> bool {
        self.cmp_eq(rhs).all()
    }

    fn _ne(self, rhs: Self) -> bool {
        self.cmp_ne(rhs).all()
    }

    fn _lt(self, rhs: Self) -> bool {
        self.cmp_lt(rhs).all()
    }

    fn _le(self, rhs: Self) -> bool {
        self.cmp_le(rhs).all()
    }

    fn _gt(self, rhs: Self) -> bool {
        self.cmp_gt(rhs).all()
    }

    fn _ge(self, rhs: Self) -> bool {
        self.cmp_ge(rhs).all()
    }
}

impl NormalOut for f64x4 {
    type Output = f64x4;
    fn _add(self, rhs: Self) -> Self::Output {
        self + rhs
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
    fn _pow(self, rhs: Self) -> Self::Output {
        self.pow_f64x4(rhs)
    }
    fn _rem(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _square(self) -> Self::Output {
        self * self
    }
    fn _abs(self) -> Self {
        self.abs()
    }
    fn _ceil(self) -> Self::Output {
        todo!()
    }
    fn _floor(self) -> Self::Output {
        todo!()
    }
    fn _sign(self) -> Self::Output {
        todo!()
    }
    fn _max(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
    fn _min(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self.round()
    }
}

impl NormalOut for i32x8 {
    type Output = i32x8;
    fn _add(self, rhs: Self) -> Self::Output {
        self + rhs
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
    fn _pow(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _rem(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _square(self) -> Self::Output {
        self * self
    }
    fn _abs(self) -> Self {
        self.abs()
    }
    fn _ceil(self) -> Self::Output {
        todo!()
    }
    fn _floor(self) -> Self::Output {
        todo!()
    }
    fn _sign(self) -> Self::Output {
        todo!()
    }
    fn _max(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
    fn _min(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self
    }
}

impl Eval for i32x8 {
    type Output = bool;

    fn _is_nan(&self) -> Self::Output {
        false
    }

    fn _is_true(&self) -> Self::Output {
        !self.cmp_eq(i32x8::splat(0)).all()
    }

    fn _is_inf(&self) -> Self::Output {
        false
    }
}
impl Cmp for i32x8 {
    fn _eq(self, rhs: Self) -> bool {
        self.cmp_eq(rhs).all()
    }

    fn _ne(self, rhs: Self) -> bool {
        self != rhs
    }

    fn _lt(self, rhs: Self) -> bool {
        self.cmp_lt(rhs).all()
    }

    fn _le(self, rhs: Self) -> bool {
        !self.cmp_gt(rhs).all()
    }

    fn _gt(self, rhs: Self) -> bool {
        self.cmp_gt(rhs).all()
    }

    fn _ge(self, rhs: Self) -> bool {
        !self.cmp_lt(rhs).all()
    }
}

impl NormalOut for i64x4 {
    type Output = i64x4;
    fn _add(self, rhs: Self) -> Self::Output {
        self + rhs
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
    fn _pow(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _rem(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _square(self) -> Self::Output {
        self * self
    }
    fn _abs(self) -> Self {
        self.abs()
    }
    fn _ceil(self) -> Self::Output {
        todo!()
    }
    fn _floor(self) -> Self::Output {
        todo!()
    }
    fn _sign(self) -> Self::Output {
        todo!()
    }
    fn _max(self, rhs: Self) -> Self::Output {
        todo!()
    }
    fn _min(self, rhs: Self) -> Self::Output {
        todo!()
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self
    }
}

impl Eval for i64x4 {
    type Output = bool;

    fn _is_nan(&self) -> Self::Output {
        false
    }

    fn _is_true(&self) -> Self::Output {
        !self.cmp_eq(i64x4::splat(0)).all()
    }

    fn _is_inf(&self) -> Self::Output {
        false
    }
}
impl Cmp for i64x4 {
    fn _eq(self, rhs: Self) -> bool {
        self.cmp_eq(rhs).all()
    }

    fn _ne(self, rhs: Self) -> bool {
        self != rhs
    }

    fn _lt(self, rhs: Self) -> bool {
        self.cmp_lt(rhs).all()
    }

    fn _le(self, rhs: Self) -> bool {
        !self.cmp_gt(rhs).all()
    }

    fn _gt(self, rhs: Self) -> bool {
        self.cmp_gt(rhs).all()
    }

    fn _ge(self, rhs: Self) -> bool {
        !self.cmp_lt(rhs).all()
    }
}

impl NormalOut for i16x16 {
    type Output = i16x16;
    fn _add(self, rhs: Self) -> Self::Output {
        self + rhs
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
    fn _pow(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _rem(self, _: Self) -> Self::Output {
        todo!()
    }
    fn _square(self) -> Self::Output {
        self * self
    }
    fn _abs(self) -> Self {
        self.abs()
    }
    fn _ceil(self) -> Self::Output {
        todo!()
    }
    fn _floor(self) -> Self::Output {
        todo!()
    }
    fn _sign(self) -> Self::Output {
        todo!()
    }
    fn _max(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
    fn _min(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self
    }
}

impl NormalOut for i8x32 {
    type Output = i8x32;
    fn _add(self, rhs: Self) -> Self::Output {
        self + rhs
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        let arr = self.as_array_ref();
        let rhs_arr = rhs.as_array_ref();
        let mut res = [0i8; 32];
        res.iter_mut()
            .zip(arr.iter().zip(rhs_arr.iter()))
            .for_each(|(r, (a, b))| {
                *r = a.wrapping_mul(*b);
            });
        i8x32::new(res)
    }
    fn _pow(self, rhs: Self) -> Self::Output {
        let arr = self.as_array_ref();
        let rhs_arr = rhs.as_array_ref();
        let mut res = [0i8; 32];
        res.iter_mut()
            .zip(arr.iter().zip(rhs_arr.iter()))
            .for_each(|(r, (a, b))| {
                *r = a.pow(*b as u32);
            });
        i8x32::new(res)
    }
    fn _rem(self, rhs: Self) -> Self::Output {
        let arr = self.as_array_ref();
        let rhs_arr = rhs.as_array_ref();
        let mut res = [0i8; 32];
        res.iter_mut()
            .zip(arr.iter().zip(rhs_arr.iter()))
            .for_each(|(r, (a, b))| {
                *r = a.rem(*b);
            });
        i8x32::new(res)
    }
    fn _square(self) -> Self::Output {
        let arr = self.as_array_ref();
        let mut res = [0i8; 32];
        res.iter_mut()
            .zip(arr.iter())
            .for_each(|(r, a)| {
                *r = a.wrapping_mul(*a);
            });
        i8x32::new(res)
    }
    fn _abs(self) -> Self {
        self.abs()
    }
    fn _ceil(self) -> Self::Output {
        self
    }
    fn _floor(self) -> Self::Output {
        self
    }
    fn _sign(self) -> Self::Output {
        let arr = self.as_array_ref();
        let mut res = [0i8; 32];
        res.iter_mut()
            .zip(arr.iter())
            .for_each(|(r, a)| {
                *r = a.signum();
            });
        i8x32::new(res)
    }
    fn _max(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
    fn _min(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self
    }
}

impl NormalOut for [bool; 32] {
    type Output = [bool; 32];
    fn _add(self, rhs: Self) -> Self::Output {
        let mut res = [false; 32];
        res.iter_mut()
            .zip(self.iter().zip(rhs.iter()))
            .for_each(|(r, (a, b))| {
                *r = *a || *b;
            });
        res
    }
    fn _sub(self, rhs: Self) -> Self::Output {
        let mut res = [false; 32];
        res.iter_mut()
            .zip(self.iter().zip(rhs.iter()))
            .for_each(|(r, (a, b))| {
                *r = *a && !*b;
            });
        res
    }
    fn _mul(self, rhs: Self) -> Self::Output {
        let mut res = [false; 32];
        res.iter_mut()
            .zip(self.iter().zip(rhs.iter()))
            .for_each(|(r, (a, b))| {
                *r = *a && *b;
            });
        res
    }
    fn _pow(self, rhs: Self) -> Self::Output {
        let mut res = [false; 32];
        res.iter_mut()
            .zip(self.iter().zip(rhs.iter()))
            .for_each(|(r, (a, b))| {
                *r = *a ^ *b;
            });
        res
    }
    fn _rem(self, rhs: Self) -> Self::Output {
        let mut res = [false; 32];
        res.iter_mut()
            .zip(self.iter().zip(rhs.iter()))
            .for_each(|(r, (a, b))| {
                *r = *a ^ *b;
            });
        res
    }
    fn _square(self) -> Self::Output {
        let mut res = [false; 32];
        res.iter_mut()
            .zip(self.iter())
            .for_each(|(r, a)| {
                *r = *a;
            });
        res
    }
    fn _abs(self) -> Self {
        self
    }
    fn _ceil(self) -> Self::Output {
        self
    }
    fn _floor(self) -> Self::Output {
        self
    }
    fn _sign(self) -> Self::Output {
        [false; 32]
    }
    fn _max(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
    fn _min(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
    fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
        todo!()
    }
    fn _round(self) -> Self::Output {
        self
    }
}

