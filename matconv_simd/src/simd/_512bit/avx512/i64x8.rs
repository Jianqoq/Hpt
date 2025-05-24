use crate::{simd::_512bit::common::{i64x8::i64x8, mask::U8MASK}, VecTrait};
use std::arch::x86_64::{
    _mm512_add_epi64, _mm512_mask_storeu_epi64, _mm512_maskz_loadu_epi64, _mm512_mullo_epi64,
    _mm512_set1_epi64,
};

impl VecTrait<i64> for i64x8 {
    #[inline(always)]
    fn splat(val: i64) -> Self {
        unsafe { Self(_mm512_set1_epi64(val)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let mul = _mm512_mullo_epi64(self.0, a.0);
            Self(_mm512_add_epi64(mul, b.0))
        }
    }

    fn partial_load(ptr: *const i64, num_elem: usize) -> Self {
        debug_assert!(num_elem <= Self::SIZE, "Cannot load more than 8 elements");
        unsafe {
            let mask = U8MASK[num_elem];
            Self(_mm512_maskz_loadu_epi64(mask, ptr))
        }
    }

    fn partial_store(self, ptr: *mut i64, num_elem: usize) {
        debug_assert!(num_elem <= Self::SIZE, "Cannot store more than 8 elements");
        unsafe {
            let mask = U8MASK[num_elem];
            _mm512_mask_storeu_epi64(ptr, mask, self.0)
        }
    }
}

impl std::ops::Add for i64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_add_epi64(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for i64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_mullo_epi64(self.0, rhs.0)) }
    }
}
