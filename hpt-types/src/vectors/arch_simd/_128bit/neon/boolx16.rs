use crate::convertion::VecConvertor;
use crate::traits::SimdCompare;
use crate::traits::VecTrait;
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_128bit::u8x16::u8x16;

use super::i8x16::i8x16;
use crate::vectors::arch_simd::_128bit::common::boolx16::boolx16;

impl VecTrait<bool> for boolx16 {
    const SIZE: usize = 16;
    type Base = bool;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        self.0.iter().map(|&x| x as u8).sum::<u8>() > 0
    }
    #[inline(always)]
    fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const bool) -> Self {
        let mut result = [false; 16];
        for i in 0..16 {
            result[i] = unsafe { *ptr.add(i) };
        }
        boolx16(result)
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
}
