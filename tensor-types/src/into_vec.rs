pub trait IntoVec<T> {
    fn into_vec(self) -> T;
}

#[cfg(target_feature = "avx2")]
mod into_vec {
    use super::IntoVec;
    use crate::vectors::_256bit::*;
    impl IntoVec<boolx32::boolx32> for u32x8::u32x8 {
        fn into_vec(self) -> boolx32::boolx32 {
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
    impl IntoVec<boolx32::boolx32> for i32x8::i32x8 {
        fn into_vec(self) -> boolx32::boolx32 {
            unreachable!()
        }
    }
    impl IntoVec<boolx32::boolx32> for i64x4::i64x4 {
        fn into_vec(self) -> boolx32::boolx32 {
            unreachable!()
        }
    }
    impl IntoVec<boolx32::boolx32> for u64x4::u64x4 {
        fn into_vec(self) -> boolx32::boolx32 {
            unreachable!()
        }
    }
}

#[cfg(all(any(target_feature = "sse2", target_feature = "neon"), not(target_feature = "avx2")))]
mod into_vec {
    use super::IntoVec;
    use crate::vectors::_128bit::*;
    impl IntoVec<boolx16::boolx16> for boolx16::boolx16 {
        fn into_vec(self) -> boolx16::boolx16 {
            unreachable!()
        }
    }
    impl IntoVec<u8x16::u8x16> for u8x16::u8x16 {
        fn into_vec(self) -> u8x16::u8x16 {
            self
        }
    }
    impl IntoVec<i8x16::i8x16> for i8x16::i8x16 {
        fn into_vec(self) -> i8x16::i8x16 {
            self
        }
    }
    impl IntoVec<u16x8::u16x8> for u16x8::u16x8 {
        fn into_vec(self) -> u16x8::u16x8 {
            self
        }
    }
    impl IntoVec<i16x8::i16x8> for i16x8::i16x8 {
        fn into_vec(self) -> i16x8::i16x8 {
            self
        }
    }
    impl IntoVec<u32x4::u32x4> for u32x4::u32x4 {
        fn into_vec(self) -> u32x4::u32x4 {
            self
        }
    }
    impl IntoVec<i32x4::i32x4> for i32x4::i32x4 {
        fn into_vec(self) -> i32x4::i32x4 {
            self
        }
    }
    impl IntoVec<i64x2::i64x2> for i64x2::i64x2 {
        fn into_vec(self) -> i64x2::i64x2 {
            self
        }
    }
    impl IntoVec<u64x2::u64x2> for u64x2::u64x2 {
        fn into_vec(self) -> u64x2::u64x2 {
            self
        }
    }
    impl IntoVec<f16x8::f16x8> for f16x8::f16x8 {
        fn into_vec(self) -> f16x8::f16x8 {
            self
        }
    }
    impl IntoVec<f32x4::f32x4> for f32x4::f32x4 {
        fn into_vec(self) -> f32x4::f32x4 {
            self
        }
    }
    impl IntoVec<f64x2::f64x2> for f64x2::f64x2 {
        fn into_vec(self) -> f64x2::f64x2 {
            self
        }
    }
    impl IntoVec<cplx32x2::cplx32x2> for cplx32x2::cplx32x2 {
        fn into_vec(self) -> cplx32x2::cplx32x2 {
            self
        }
    }
    impl IntoVec<cplx64x1::cplx64x1> for cplx64x1::cplx64x1 {
        fn into_vec(self) -> cplx64x1::cplx64x1 {
            self
        }
    }
}
