//! This crate provides a set of iterators for tensors.

#![deny(missing_docs)]

use par_strided::{par_strided_simd::ParStridedSimd, ParStrided};
use par_strided_mut::{par_strided_map_mut_simd::ParStridedMutSimd, ParStridedMut};
use strided::{strided_simd::StridedSimd, Strided};
use strided_mut::{simd_imports::StridedMutSimd, StridedMut};
use tensor_traits::{CommonBounds, TensorInfo};

/// A module contains all the iterator traits
pub mod iterator_traits;
/// A module contains simd/non-simd parallel strided iterator
pub mod par_strided;
/// A module contains simd/non-simd parallel strided fold iterator
pub mod par_strided_fold;
/// A module contains simd/non-simd parallel strided map iterator
pub mod par_strided_map;
/// A module contains mutable simd/non-simd parallel strided map iterator
pub mod par_strided_map_mut;
/// A module contains mutable simd/non-simd parallel strided iterator
pub mod par_strided_mut;
/// A module contains simd/non-simd prallel strided zip iterator
pub mod par_strided_zip;
/// A module contains shape manipulation functions
pub mod shape_manipulate;
/// A module contains simd/non-simd strided iterator
pub mod strided;
/// A module contains simd/non-simd strided map iterator
pub mod strided_map;
/// A module contains mutable simd/non-simd strided map iterator
pub mod strided_map_mut;
/// A module contains simd/non-simd strided map zip iterator
pub mod strided_mut;
/// A module contains simd/non-simd strided zip iterator
pub mod strided_zip;

mod with_simd;

/// A trait for converting a tensor into an iterator.
pub trait TensorIterator<'a, T: CommonBounds>
where
    Self: TensorInfo<T> + 'a,
    &'a Self: TensorInfo<T>,
    &'a mut Self: TensorInfo<T>,
{
    /// Convert the tensor into a strided iterator.
    ///
    /// strided iterator is a single-threaded iterator
    fn iter(&'a self) -> Strided<T> {
        Strided::new(self)
    }

    /// Convert the tensor into a mutable strided iterator.
    ///
    /// strided iterator is a single-threaded iterator
    fn iter_mut(&'a mut self) -> StridedMut<'a, T> {
        StridedMut::new(self)
    }

    /// Convert the tensor into a strided simd iterator.
    ///
    /// strided simd iterator is a single-threaded simd iterator
    fn iter_simd(&'a self) -> StridedSimd<T> {
        StridedSimd::new(self)
    }

    /// Convert the tensor into a mutable strided simd iterator.
    ///
    /// strided simd iterator is a single-threaded simd iterator
    fn iter_mut_simd(&'a self) -> StridedMutSimd<'a, T> {
        StridedMutSimd::new(self)
    }

    /// Convert the tensor into a parallel strided simd iterator.
    ///
    /// parallel strided simd iterator is a multi-threaded simd iterator
    fn par_iter_simd(&'a self) -> ParStridedSimd<T> {
        ParStridedSimd::new(self)
    }

    /// Convert the tensor into a mutable parallel strided simd iterator.
    ///
    /// parallel strided simd iterator is a multi-threaded simd iterator
    fn par_iter_mut_simd(&'a mut self) -> ParStridedMutSimd<'a, T> {
        ParStridedMutSimd::new(self)
    }

    /// Convert the tensor into a parallel strided iterator.
    ///
    /// parallel strided iterator is a multi-threaded iterator
    fn par_iter(&'a self) -> ParStrided<T> {
        ParStrided::new(self)
    }

    /// Convert the tensor into a mutable parallel strided iterator.
    ///
    /// parallel strided iterator is a multi-threaded iterator
    fn par_iter_mut(&'a mut self) -> ParStridedMut<'a, T> {
        ParStridedMut::new(self)
    }
}
