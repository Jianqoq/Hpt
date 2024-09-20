use par_strided::{par_strided_simd::ParStridedSimd, ParStrided};
use par_strided_mut::{par_strided_map_mut_simd::ParStridedMutSimd, ParStridedMut};
use strided::{strided_simd::StridedSimd, Strided};
use strided_mut::{simd_imports::StridedMutSimd, StridedMut};
use tensor_traits::{CommonBounds, TensorInfo};

pub mod iterator_traits;
pub mod par_strided;
pub mod par_strided_fold;
pub mod par_strided_map;
pub mod par_strided_map_mut;
pub mod par_strided_mut;
pub mod par_strided_zip;
pub mod strided;
pub mod strided_map;
pub mod strided_map_mut;
pub mod strided_mut;
pub mod strided_zip;
mod with_simd;

pub trait TensorIterator<'a, T: CommonBounds>
where
    Self: TensorInfo<T> + 'a,
    &'a Self: TensorInfo<T>,
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
    fn iter_mut(&'a self) -> StridedMut<T> {
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
    fn iter_mut_simd(&'a self) -> StridedMutSimd<T> {
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
    fn par_iter_mut_simd(&'a self) -> ParStridedMutSimd<T> {
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
    fn par_iter_mut(&'a self) -> ParStridedMut<T> {
        ParStridedMut::new(self)
    }
}
