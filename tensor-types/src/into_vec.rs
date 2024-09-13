#[cfg(target_feature = "avx2")]
use crate::vectors::_256bit::*;
#[cfg(all(any(target_feature = "sse2", target_feature = "neon"), not(target_feature = "avx2")))]
use crate::vectors::_128bit::*;
pub trait IntoVec<T> {
    fn into_vec(self) -> T;
}

impl IntoVec<boolx32::boolx32> for u32x8::u32x8 {
    fn into_vec(self) -> boolx32::boolx32 {
        unreachable!()
    }
}
#[cfg(all(any(target_feature = "sse2", target_feature = "neon"), not(target_feature = "avx2")))]
impl IntoVec<boolx16::boolx16> for u32x8::u32x8 {
    fn into_vec(self) -> boolx16::boolx16 {
        unreachable!()
    }
}

impl IntoVec<u32x8::u32x8> for u32x8::u32x8 {
    fn into_vec(self) -> u32x8::u32x8 {
        self
    }
}


impl IntoVec<i64x4::i64x4> for i64x4::i64x4 {
    fn into_vec(self) -> i64x4::i64x4 {
        self
    }
}
