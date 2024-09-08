use crate::vectors::_256bit::*;

use crate::vectors::_256bit::boolx32::boolx32;

pub trait IntoVec<T> {
    fn into_vec(self) -> T;
}

impl IntoVec<boolx32> for u32x8::u32x8 {
    fn into_vec(self) -> boolx32 {
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
