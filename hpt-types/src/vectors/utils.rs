use super::traits::VecTrait;
use crate::{dtype::TypeCommon, type_promote::NormalOut};

/// impl bit logic for simd vector
#[macro_export]
macro_rules! impl_std_simd_bit_logic {
    ($ty: ident) => {
        impl std::ops::BitAnd for $ty {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self::Output {
                $ty(self.0 & rhs.0)
            }
        }
        impl std::ops::BitOr for $ty {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self::Output {
                $ty(self.0 | rhs.0)
            }
        }
        impl std::ops::BitXor for $ty {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                $ty(self.0 ^ rhs.0)
            }
        }
        impl std::ops::Not for $ty {
            type Output = Self;
            fn not(self) -> Self::Output {
                $ty(!self.0)
            }
        }
        impl std::ops::Shl for $ty {
            type Output = Self;
            fn shl(self, rhs: Self) -> Self {
                $ty(self.0 << rhs.0)
            }
        }
        impl std::ops::Shr for $ty {
            type Output = Self;
            fn shr(self, rhs: Self) -> Self {
                $ty(self.0 >> rhs.0)
            }
        }
    };
}

/// sum a vector to a scalar
#[inline(always)]
pub fn vec_sum<T: TypeCommon + NormalOut<T, Output = T>>(vec: T::Vec) -> T {
    let mut sum = T::ZERO;
    for i in 0..T::Vec::SIZE {
        sum = sum._add(vec.extract(i));
    }
    sum
}

/// sum an array to a scalar with simd
#[inline(always)]
pub fn array_vec_sum<T: TypeCommon + NormalOut<T, Output = T> + Copy>(array: &[T]) -> T {
    let remain = array.len() % T::Vec::SIZE;
    let vecs = unsafe {
        std::slice::from_raw_parts(
            array as *const _ as *const T::Vec,
            array.len() / T::Vec::SIZE,
        )
    };
    let mut sum_vec = T::Vec::splat(T::ZERO);
    for vec in vecs.iter() {
        sum_vec = sum_vec._add(vec.read_unaligned());
    }
    let mut sum = T::ZERO;
    for i in array.iter().skip(array.len() - remain) {
        sum = sum._add(*i);
    }
    for i in 0..T::Vec::SIZE {
        sum = sum._add(sum_vec.extract(i));
    }
    sum
}

/// sum an array to a scalar with simd
#[inline(always)]
pub fn array_vec_reduce<T: TypeCommon + NormalOut<T, Output = T> + Copy>(
    array: &[T],
    init: T,
    vec_op: impl Fn(T::Vec, T::Vec) -> T::Vec,
    scalar_op: impl Fn(T, T) -> T,
    vec_reduce_op: impl Fn(T, T) -> T,
) -> T {
    let remain = array.len() % T::Vec::SIZE;
    let vecs = array as *const _ as *const T::Vec;
    let mut red_vec = T::Vec::splat(init);
    for i in 0..array.len() / T::Vec::SIZE {
        red_vec = vec_op(red_vec, unsafe { vecs.add(i).read_unaligned() });
    }
    let mut red = init;
    for i in array.iter().skip(array.len() - remain) {
        red = scalar_op(red, *i);
    }
    for i in 0..T::Vec::SIZE {
        red = vec_reduce_op(red, red_vec.extract(i));
    }
    red
}
