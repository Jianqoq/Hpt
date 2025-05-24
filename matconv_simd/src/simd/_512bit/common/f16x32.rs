use crate::VecTrait;
use crate::simd::_512bit::avx512::f16x32::f32x16_to_f16x16;
use crate::simd::_512bit::common::f32x16::f32x16;

/// a vector of 16 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct f16x32(pub(crate) [half::f16; 32]);

impl VecTrait<half::f16> for f16x32 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x16; 2] = unsafe { std::mem::transmute(self.to_2_f32vec()) };
        let [a0, a1]: [f32x16; 2] = unsafe { std::mem::transmute(a.to_2_f32vec()) };
        let [b0, b1]: [f32x16; 2] = unsafe { std::mem::transmute(b.to_2_f32vec()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        let res0 = f32x16_to_f16x16(res0);
        let res1 = f32x16_to_f16x16(res1);
        unsafe { std::mem::transmute([res0, res1]) }
    }
    #[inline(always)]
    fn splat(val: half::f16) -> f16x32 {
        f16x32([val; 32])
    }
    #[inline(always)]
    fn partial_load(ptr: *const half::f16, num_elem: usize) -> Self {
        let mut result = f16x32::default();
        unsafe { std::ptr::copy_nonoverlapping(ptr, result.0.as_mut_ptr(), num_elem) };
        result
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut half::f16, num_elem: usize) {
        unsafe { std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, num_elem) };
    }
}

impl std::ops::Add for f16x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        f16x32::from_2_f32vec([low_add, high_add])
    }
}

impl std::ops::Mul for f16x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_mul = x0 * y0;
        let high_mul = x1 * y1;
        f16x32::from_2_f32vec([low_mul, high_mul])
    }
}
