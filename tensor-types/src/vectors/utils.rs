use crate::{ dtype::TypeCommon, type_promote::NormalOut };
use super::traits::{ VecCommon, VecTrait };
use crate::vectors::traits::Init;

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
pub fn array_vec_sum<T: TypeCommon + VecCommon + NormalOut<T, Output = T> + Copy>(
    array: &[T]
) -> T {
    let remain = array.len() % T::Vec::SIZE;
    let vecs = unsafe {
        std::slice::from_raw_parts(array as *const _ as *const T::Vec, array.len() / T::Vec::SIZE)
    };
    let mut sum_vec = T::Vec::splat(T::ZERO);
    for vec in vecs.iter() {
        sum_vec = sum_vec._add(vec.read_unaligned());
    }
    let mut sum = T::ZERO;
    for i in array.len() - remain..array.len() {
        sum = sum._add(array[i]);
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
    vec_op: impl Fn(T::Vec, T::Vec) -> T::Vec,
    scalar_op: impl Fn(T, T) -> T
) -> T {
    let remain = array.len() % T::Vec::SIZE;
    let vecs = unsafe {
        std::slice::from_raw_parts(array as *const _ as *const T::Vec, array.len() / T::Vec::SIZE)
    };
    let mut sum_vec = T::Vec::splat(T::ZERO);
    for vec in vecs.iter() {
        sum_vec = vec_op(sum_vec, vec.read_unaligned());
    }
    let mut sum = T::ZERO;
    for i in array.len() - remain..array.len() {
        sum = scalar_op(sum, array[i]);
    }
    for i in 0..T::Vec::SIZE {
        sum = sum._add(sum_vec.extract(i));
    }
    sum
}
