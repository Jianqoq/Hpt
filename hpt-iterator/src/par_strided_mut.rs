use crate::{
    iterator_traits::{IterGetSet, ParStridedHelper, ParStridedIteratorZip, ShapeManipulator},
    par_strided::ParStrided,
    shape_manipulate::{par_expand, par_reshape, par_transpose},
};
use hpt_common::shape::shape::Shape;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use rayon::iter::{
    plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};
use std::sync::Arc;

/// A module for parallel mutable strided iterator.
pub mod par_strided_map_mut_simd {
    use crate::{
        iterator_traits::{IterGetSetSimd, ParStridedIteratorSimd, ParStridedIteratorSimdZip},
        par_strided::par_strided_simd::ParStridedSimd,
    };
    use crate::{CommonBounds, TensorInfo};
    use hpt_common::{shape::shape::Shape, utils::pointer::Pointer, utils::simd_ref::MutVec};
    use hpt_types::dtype::TypeCommon;
    use hpt_types::traits::VecTrait;
    use rayon::iter::{
        plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
        ParallelIterator,
    };
    use std::sync::Arc;
    /// A parallel mutable SIMD-optimized strided iterator over tensor elements.
    ///
    /// This struct provides mutable access to tensor elements with strided access patterns optimized
    /// for parallel and SIMD (Single Instruction, Multiple Data) processing. It leverages Rayon for
    /// concurrent execution and ensures efficient traversal and modification of tensor data based on
    /// their strides.
    pub struct ParStridedMutSimd<'a, T: TypeCommon + Send + Copy + Sync> {
        /// The base parallel SIMD-optimized strided iterator.
        pub(crate) base: ParStridedSimd<T>,
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, T: CommonBounds> ParStridedMutSimd<'a, T> {
        /// Creates a new `ParStridedMutSimd` instance from a given tensor.
        ///
        /// This constructor initializes the `ParStridedMutSimd` iterator by wrapping the provided tensor
        /// in a `ParStridedSimd` instance. It sets up the necessary data structures for parallel and SIMD
        /// optimized iteration.
        ///
        /// # Arguments
        ///
        /// * `tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
        ///
        /// # Returns
        ///
        /// A new instance of `ParStridedMutSimd` initialized with the provided tensor.
        pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
            ParStridedMutSimd {
                base: ParStridedSimd::new(tensor),
                phantom: std::marker::PhantomData,
            }
        }
    }

    impl<'a, T: CommonBounds> ParStridedIteratorSimdZip for ParStridedMutSimd<'a, T> {}
    impl<'a, T: CommonBounds> ParStridedIteratorSimd for ParStridedMutSimd<'a, T> {}

    impl<'a, T> ParallelIterator for ParStridedMutSimd<'a, T>
    where
        T: CommonBounds,
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

    impl<'a, T> UnindexedProducer for ParStridedMutSimd<'a, T>
    where
        T: CommonBounds,
        T::Vec: Send,
    {
        type Item = &'a mut T;

        fn split(self) -> (Self, Option<Self>) {
            let (a, b) = self.base.split();
            (
                ParStridedMutSimd {
                    base: a,
                    phantom: std::marker::PhantomData,
                },
                b.map(|x| ParStridedMutSimd {
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

    impl<'a, T: 'a> IterGetSetSimd for ParStridedMutSimd<'a, T>
    where
        T: CommonBounds,
        T::Vec: Send,
    {
        type Item = &'a mut T;

        type SimdItem
            = MutVec<'a, T::Vec>
        where
            Self: 'a;

        fn set_end_index(&mut self, end_index: usize) {
            self.base.set_end_index(end_index);
        }

        fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.base.set_intervals(intervals);
        }

        fn set_strides(&mut self, strides: hpt_common::strides::strides::Strides) {
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

        fn strides(&self) -> &hpt_common::strides::strides::Strides {
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
            unsafe {
                self.base
                    .ptr
                    .get_ptr()
                    .add(index * (self.base.last_stride as usize))
                    .as_mut()
                    .unwrap()
            }
        }

        #[inline(always)]
        fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem {
            unsafe {
                let ptr = self.base.ptr.get_ptr().add(index * T::Vec::SIZE) as *mut T::Vec;
                #[cfg(feature = "bound_check")]
                return MutVec::new(Pointer::new(ptr, T::Vec::SIZE as i64));
                #[cfg(not(feature = "bound_check"))]
                return MutVec::new(Pointer::new(ptr));
            }
        }

        fn all_last_stride_one(&self) -> bool {
            self.base.all_last_stride_one()
        }

        fn lanes(&self) -> Option<usize> {
            self.base.lanes()
        }
    }
}

/// A parallel mutable strided iterator over tensor elements.
///
/// This struct provides mutable access to tensor elements with strided access patterns optimized
/// for parallel processing. It leverages Rayon for concurrent execution, allowing efficient traversal
/// and modification of tensor data based on their strides.
pub struct ParStridedMut<'a, T> {
    /// The base parallel strided iterator.
    pub(crate) base: ParStrided<T>,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: CommonBounds> ParStridedHelper for ParStridedMut<'a, T> {
    fn _set_last_strides(&mut self, stride: i64) {
        self.base._set_last_strides(stride);
    }

    fn _set_strides(&mut self, strides: hpt_common::strides::strides::Strides) {
        self.base._set_strides(strides);
    }

    fn _set_shape(&mut self, shape: Shape) {
        self.base._set_shape(shape);
    }

    fn _layout(&self) -> &hpt_common::layout::layout::Layout {
        self.base._layout()
    }

    fn _set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.base._set_intervals(intervals);
    }

    fn _set_end_index(&mut self, end_index: usize) {
        self.base._set_end_index(end_index);
    }
}

impl<'a, T: CommonBounds> ShapeManipulator for ParStridedMut<'a, T> {
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
        par_reshape(self, shape)
    }

    fn transpose<AXIS: Into<hpt_common::axis::axis::Axis>>(self, axes: AXIS) -> Self {
        par_transpose(self, axes)
    }

    fn expand<S: Into<Shape>>(self, shape: S) -> Self {
        par_expand(self, shape)
    }
}

impl<'a, T: CommonBounds> ParStridedIteratorZip for ParStridedMut<'a, T> {}

impl<'a, T: CommonBounds> ParStridedMut<'a, T> {
    /// Creates a new `ParStridedMut` instance from a given tensor.
    ///
    /// This constructor initializes the `ParStridedMut` iterator by wrapping the provided tensor
    /// in a `ParStrided` instance. It sets up the necessary data structures for parallel iteration.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
    ///
    /// # Returns
    ///
    /// A new instance of `ParStridedMut` initialized with the provided tensor.
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        ParStridedMut {
            base: ParStrided::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> ParallelIterator for ParStridedMut<'a, T>
where
    T: CommonBounds,
{
    type Item = &'a mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }
}

impl<'a, T> UnindexedProducer for ParStridedMut<'a, T>
where
    T: CommonBounds,
{
    type Item = &'a mut T;

    fn split(self) -> (Self, Option<Self>) {
        let (a, b) = self.base.split();
        (
            ParStridedMut {
                base: a,
                phantom: std::marker::PhantomData,
            },
            b.map(|x| ParStridedMut {
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

impl<'a, T: 'a> IterGetSet for ParStridedMut<'a, T>
where
    T: CommonBounds,
{
    type Item = &'a mut T;

    fn set_end_index(&mut self, end_index: usize) {
        self.base.set_end_index(end_index);
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.base.set_intervals(intervals);
    }

    fn set_strides(&mut self, strides: hpt_common::strides::strides::Strides) {
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

    fn strides(&self) -> &hpt_common::strides::strides::Strides {
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
        unsafe {
            self.base
                .ptr
                .get_ptr()
                .add(index * (self.base.last_stride as usize))
                .as_mut()
                .unwrap()
        }
    }
}
