use crate::traits::VecTrait;
use crate::vectors::arch_simd::_128bit::common::boolx16::boolx16;

impl VecTrait<bool> for boolx16 {
    const SIZE: usize = 16;
    type Base = bool;
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
}
