use std::sync::Arc;

use hpt_common::{shape::shape::Shape, strides::strides::Strides};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use rayon::iter::{
    plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};

use crate::{
    iterator_traits::IterGetSet, par_strided_mut::ParStridedMut, par_strided_zip::ParStridedZip,
};

/// A module for parallel strided mutable map iterator.
pub mod par_strided_map_mut_simd {
    use std::sync::Arc;

    use hpt_common::{shape::shape::Shape, strides::strides::Strides, utils::simd_ref::MutVec};
    use hpt_traits::{CommonBounds, TensorInfo};
    use hpt_types::dtype::TypeCommon;
    use rayon::iter::{
        plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
        ParallelIterator,
    };

    use crate::{
        iterator_traits::IterGetSetSimd,
        par_strided_mut::par_strided_map_mut_simd::ParStridedMutSimd,
        par_strided_zip::par_strided_zip_simd::ParStridedZipSimd,
    };

    /// A parallel mutable SIMD-optimized map iterator.
    ///
    /// This struct provides mutable access to tensor elements with strided access patterns optimized
    pub struct ParStridedMapMutSimd<'a, T>
    where
        T: TypeCommon + Send + Copy + Sync,
    {
        /// The underlying parallel SIMD-optimized strided iterator.
        pub(crate) base: ParStridedMutSimd<'a, T>,
        /// Phantom data to associate the lifetime `'a` with the struct.
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, T> ParStridedMapMutSimd<'a, T>
    where
        T: CommonBounds,
        T::Vec: Send,
    {
        /// Creates a new `ParStridedMapMutSimd` instance from a given result tensor.
        ///
        /// This constructor initializes the `ParStridedMapMutSimd` iterator by wrapping the provided result
        /// tensor in a `ParStridedMutSimd` instance. It sets up the necessary data structures for parallel and SIMD
        /// optimized iteration.
        ///
        /// # Type Parameters
        ///
        /// * `U` - The type of the tensor to accumulate results into. Must implement `TensorInfo<T>`.
        ///
        /// # Arguments
        ///
        /// * `res_tensor` - The tensor implementing the `TensorInfo<T>` trait to accumulate results into.
        ///
        /// # Returns
        ///
        /// A new instance of `ParStridedMapMutSimd` initialized with the provided result tensor.
        pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
            ParStridedMapMutSimd {
                base: ParStridedMutSimd::new(res_tensor),
                phantom: std::marker::PhantomData,
            }
        }
        /// Combines this `ParStridedMapMutSimd` iterator with another SIMD-optimized iterator, enabling simultaneous parallel iteration.
        ///
        /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
        /// iterate over tensors with compatible shapes. It calculates the appropriate iteration intervals based
        /// on the new broadcasted shape and configures both iterators accordingly. Finally, it returns a new
        /// `ParStridedZipSimd` instance that allows for synchronized parallel iteration over the combined iterators.
        ///
        /// **Note:** This implementation leverages Rayon for parallel execution and assumes that the iterators
        /// support SIMD optimizations.
        ///
        /// # Type Parameters
        ///
        /// * `C` - The type of the other iterator to zip with. Must implement `UnindexedProducer`, `IterGetSetSimd`, and `ParallelIterator`.
        ///
        /// # Arguments
        ///
        /// * `self` - The `ParStridedMapMutSimd` iterator instance.
        /// * `other` - The other iterator to zip with.
        ///
        /// # Returns
        ///
        /// A new `ParStridedZipSimd` instance that combines `self` and `other` for synchronized parallel iteration over both iterators.
        pub fn zip<C>(self, other: C) -> ParStridedZipSimd<'a, Self, C>
        where
            C: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator,
            <C as IterGetSetSimd>::Item: Send,
            T::Vec: Send,
        {
            ParStridedZipSimd::new(self, other)
        }
    }

    impl<'a, T> ParallelIterator for ParStridedMapMutSimd<'a, T>
    where
        T: 'a + CommonBounds,
        T::Vec: Send,
    {
        type Item = &'a mut T;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge_unindexed(self, consumer)
        }
    }

    impl<'a, T> UnindexedProducer for ParStridedMapMutSimd<'a, T>
    where
        T: 'a + CommonBounds,
        T::Vec: Send,
    {
        type Item = &'a mut T;

        fn split(self) -> (Self, Option<Self>) {
            let (a, b) = self.base.split();
            (
                ParStridedMapMutSimd {
                    base: a,
                    phantom: std::marker::PhantomData,
                },
                b.map(|x| ParStridedMapMutSimd {
                    base: x,
                    phantom: std::marker::PhantomData,
                }),
            )
        }

        fn fold_with<F>(self, folder: F) -> F
        where
            F: Folder<Self::Item>,
        {
            folder
        }
    }

    impl<'a, T: 'a + CommonBounds> IterGetSetSimd for ParStridedMapMutSimd<'a, T>
    where
        T::Vec: Send,
    {
        type Item = &'a mut T;

        type SimdItem = MutVec<'a, T::Vec>;

        fn set_end_index(&mut self, end_index: usize) {
            self.base.set_end_index(end_index);
        }

        fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.base.set_intervals(intervals);
        }

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

        fn layout(&self) -> &hpt_common::layout::layout::Layout {
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
            self.base.next_simd();
        }

        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            self.base.inner_loop_next(index)
        }

        fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem {
            self.base.inner_loop_next_simd(index)
        }

        fn all_last_stride_one(&self) -> bool {
            self.base.all_last_stride_one()
        }

        fn lanes(&self) -> Option<usize> {
            self.base.lanes()
        }
    }
}

/// A parallel mutable map iterator.
///
/// This struct provides mutable access to tensor elements with strided access
pub struct ParStridedMapMut<'a, T>
where
    T: Copy,
{
    /// The base parallel strided mutable iterator.
    pub(crate) base: ParStridedMut<'a, T>,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> ParStridedMapMut<'a, T>
where
    T: CommonBounds,
{
    /// Creates a new `ParStridedMapMut` instance from a given result tensor.
    ///
    /// This constructor initializes the `ParStridedMapMut` iterator by wrapping the provided result
    /// tensor in a `ParStridedMut` instance. It sets up the necessary data structures for parallel iteration.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The type of the tensor to accumulate results into. Must implement `TensorInfo<T>`.
    ///
    /// # Arguments
    ///
    /// * `res_tensor` - The tensor implementing the `TensorInfo<T>` trait to accumulate results into.
    ///
    /// # Returns
    ///
    /// A new instance of `ParStridedMapMut` initialized with the provided result tensor.
    pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
        ParStridedMapMut {
            base: ParStridedMut::new(res_tensor),
            phantom: std::marker::PhantomData,
        }
    }
    /// Combines this `ParStridedMapMut` iterator with another iterator, enabling simultaneous parallel iteration.
    ///
    /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
    /// iterate over tensors with compatible shapes. It calculates the appropriate iteration intervals based
    /// on the new broadcasted shape and configures both iterators accordingly. Finally, it returns a new
    /// `ParStridedZip` instance that allows for synchronized parallel iteration over the combined iterators.
    ///
    /// **Note:** This implementation leverages Rayon for parallel execution.
    ///
    /// # Type Parameters
    ///
    /// * `C` - The type of the other iterator to zip with. Must implement `UnindexedProducer`, `IterGetSet`, and `ParallelIterator`.
    ///
    /// # Arguments
    ///
    /// * `self` - The `ParStridedMapMut` iterator instance.
    /// * `other` - The other iterator to zip with.
    ///
    /// # Returns
    ///
    /// A new `ParStridedZip` instance that combines `self` and `other` for synchronized parallel iteration over both iterators.
    pub fn zip<C>(self, other: C) -> ParStridedZip<'a, Self, C>
    where
        C: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
        <C as IterGetSet>::Item: Send,
    {
        ParStridedZip::new(self, other)
    }
}

impl<'a, T> ParallelIterator for ParStridedMapMut<'a, T>
where
    T: 'a + CommonBounds,
{
    type Item = &'a mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }
}

impl<'a, T> UnindexedProducer for ParStridedMapMut<'a, T>
where
    T: 'a + CommonBounds,
{
    type Item = &'a mut T;

    fn split(self) -> (Self, Option<Self>) {
        let (a, b) = self.base.split();
        (
            ParStridedMapMut {
                base: a,
                phantom: std::marker::PhantomData,
            },
            b.map(|x| ParStridedMapMut {
                base: x,
                phantom: std::marker::PhantomData,
            }),
        )
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        folder
    }
}

impl<'a, T: 'a + CommonBounds> IterGetSet for ParStridedMapMut<'a, T> {
    type Item = &'a mut T;

    fn set_end_index(&mut self, end_index: usize) {
        self.base.set_end_index(end_index);
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.base.set_intervals(intervals);
    }

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
    fn layout(&self) -> &hpt_common::layout::layout::Layout {
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
