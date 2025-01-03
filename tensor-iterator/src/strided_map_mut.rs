use std::sync::Arc;
use tensor_common::{shape::shape::Shape, strides::Strides};
use tensor_traits::tensor::{CommonBounds, TensorInfo};
use tensor_types::dtype::TypeCommon;

use crate::{
    iterator_traits::{IterGetSet, StridedIterator},
    par_strided_mut::ParStridedMut,
    strided_zip::StridedZip,
};

/// A module for mutable mapped strided iterator.
pub mod strided_map_mut_simd {
    use std::sync::Arc;

    use tensor_common::{shape::shape::Shape, strides::Strides};
    use tensor_traits::{CommonBounds, TensorInfo};
    use tensor_types::dtype::TypeCommon;

    use crate::{
        iterator_traits::{IterGetSetSimd, StridedIteratorSimd},
        par_strided_mut::par_strided_map_mut_simd::ParStridedMutSimd,
        strided_zip::strided_zip_simd::StridedZipSimd,
    };

    /// A SIMD-optimized mutable mapped strided iterator over tensor elements.
    ///
    /// This struct provides mutable access to tensor elements with SIMD optimizations,
    /// allowing for efficient parallel processing of tensor data.
    pub struct StridedMapMutSimd<'a, T>
    where
        T: Copy + TypeCommon + Send + Sync,
    {
        /// The underlying parallel SIMD-optimized strided iterator.
        pub(crate) base: ParStridedMutSimd<'a, T>,
        /// Phantom data to associate the lifetime `'a` with the struct.
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }
    impl<'a, T> StridedMapMutSimd<'a, T>
    where
        T: CommonBounds,
        T::Vec: Send,
    {
        /// Creates a new `StridedMapMutSimd` instance from a given tensor.
        ///
        /// # Arguments
        ///
        /// * `res_tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
        ///
        /// # Returns
        ///
        /// A new instance of `StridedMapMutSimd` initialized with the provided tensor.
        pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
            StridedMapMutSimd {
                base: ParStridedMutSimd::new(res_tensor),
                phantom: std::marker::PhantomData,
            }
        }
        /// Combines this `StridedMapMutSimd` iterator with another SIMD-optimized iterator, enabling simultaneous iteration.
        ///
        /// This method zips together `self` and `other` into a `StridedZipSimd` iterator, allowing for synchronized
        /// iteration over both iterators. This is particularly useful for operations that require processing
        /// elements from two tensors in parallel.
        ///
        /// # Arguments
        ///
        /// * `other` - The other iterator to zip with. It must implement the `IterGetSetSimd` trait, and
        ///             its associated `Item` type must be `Send`.
        ///
        /// # Returns
        ///
        /// A `StridedZipSimd` instance that zips together `self` and `other`, enabling synchronized
        /// iteration over their elements.
        pub(crate) fn zip<C>(self, other: C) -> StridedZipSimd<'a, Self, C>
        where
            C: 'a + IterGetSetSimd,
            <C as IterGetSetSimd>::Item: Send,
        {
            StridedZipSimd::new(self, other)
        }
    }
    impl<'a, T> StridedIteratorSimd for StridedMapMutSimd<'a, T> where T: 'a + CommonBounds {}
    impl<'a, T: 'a + CommonBounds> IterGetSetSimd for StridedMapMutSimd<'a, T>
    where
        T::Vec: Send,
    {
        type Item = &'a mut T;
        type SimdItem = &'a mut T::Vec;

        fn set_end_index(&mut self, _: usize) {}

        fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {}

        fn set_strides(&mut self, strides: Strides) {
            self.base.set_strides(strides);
        }

        fn set_shape(&mut self, shape: Shape) {
            self.base.set_shape(shape);
        }

        fn set_prg(&mut self, prg: Vec<i64>) {
            self.base.set_prg(prg);
        }

        fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
            self.base.intervals()
        }

        fn strides(&self) -> &Strides {
            self.base.strides()
        }

        fn shape(&self) -> &Shape {
            self.base.shape()
        }

        fn layout(&self) -> &tensor_common::layout::Layout {
            self.base.layout()
        }

        fn broadcast_set_strides(&mut self, shape: &Shape) {
            self.base.broadcast_set_strides(shape);
        }

        fn outer_loop_size(&self) -> usize {
            self.base.outer_loop_size()
        }

        fn inner_loop_size(&self) -> usize {
            self.base.inner_loop_size()
        }

        fn next(&mut self) {
            self.base.next();
        }

        fn next_simd(&mut self) {
            todo!()
        }

        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            self.base.inner_loop_next(index)
        }

        fn inner_loop_next_simd(&mut self, _: usize) -> Self::SimdItem {
            todo!()
        }

        fn all_last_stride_one(&self) -> bool {
            todo!()
        }

        fn lanes(&self) -> Option<usize> {
            todo!()
        }
    }
}

/// A `non` SIMD-optimized mutable mapped strided iterator over tensor elements.
///
/// This struct provides mutable access to tensor elements,
pub struct StridedMapMut<'a, T>
where
    T: Copy + TypeCommon,
{
    /// The underlying parallel mutable strided iterator.
    pub(crate) base: ParStridedMut<'a, T>,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> StridedMapMut<'a, T>
where
    T: CommonBounds,
    T::Vec: Send,
{
    /// Creates a new `StridedMapMut` instance from a given tensor.
    ///
    /// # Arguments
    ///
    /// * `res_tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
    ///
    /// # Returns
    ///
    /// A new instance of `StridedMapMut` initialized with the provided tensor.
    pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
        StridedMapMut {
            base: ParStridedMut::new(res_tensor),
            phantom: std::marker::PhantomData,
        }
    }

    /// Combines this `StridedMapMut` iterator with another iterator, enabling simultaneous iteration.
    ///
    /// This method zips together `self` and `other` into a `StridedZip` iterator, allowing for synchronized
    /// iteration over both iterators. This is particularly useful for operations that require processing
    /// elements from two tensors in parallel, such as element-wise arithmetic operations.
    ///
    /// # Arguments
    ///
    /// * `other` - The other iterator to zip with. It must implement the `IterGetSet` trait, and
    ///             its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A `StridedZip` instance that encapsulates both `self` and `other`, allowing for synchronized
    /// iteration over their elements.
    pub fn zip<C>(self, other: C) -> StridedZip<'a, Self, C>
    where
        C: 'a + IterGetSet,
        <C as IterGetSet>::Item: Send,
    {
        StridedZip::new(self, other)
    }
}

impl<'a, T> StridedIterator for StridedMapMut<'a, T> where T: 'a + CommonBounds {}

impl<'a, T: 'a + CommonBounds> IterGetSet for StridedMapMut<'a, T>
where
    T::Vec: Send,
{
    type Item = &'a mut T;

    fn set_end_index(&mut self, _: usize) {}

    fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {}

    fn set_strides(&mut self, strides: Strides) {
        self.base.set_strides(strides);
    }

    fn set_shape(&mut self, shape: Shape) {
        self.base.set_shape(shape);
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.base.set_prg(prg);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        self.base.intervals()
    }

    fn strides(&self) -> &Strides {
        self.base.strides()
    }

    fn shape(&self) -> &Shape {
        self.base.shape()
    }

    fn layout(&self) -> &tensor_common::layout::Layout {
        self.base.layout()
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        self.base.broadcast_set_strides(shape);
    }

    fn outer_loop_size(&self) -> usize {
        self.base.outer_loop_size()
    }

    fn inner_loop_size(&self) -> usize {
        self.base.inner_loop_size()
    }

    fn next(&mut self) {
        self.base.next();
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        self.base.inner_loop_next(index)
    }
}
