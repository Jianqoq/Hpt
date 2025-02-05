use crate::{
    iterator_traits::{IterGetSet, StridedIterator, StridedIteratorZip},
    strided::Strided,
    strided_zip::StridedZip,
};
use std::sync::Arc;
use hpt_common::{shape::shape::Shape, shape::shape_utils::predict_broadcast_shape};
use hpt_traits::tensor::{CommonBounds, TensorInfo};

/// Module containing SIMD-optimized implementations for strided mutability.
pub mod simd_imports {
    use crate::{
        iterator_traits::{IterGetSetSimd, StridedIteratorSimd, StridedSimdIteratorZip},
        strided::strided_simd::StridedSimd,
    };
    use std::sync::Arc;
    use hpt_common::shape::shape::Shape;
    use hpt_traits::{CommonBounds, TensorInfo};
    use hpt_types::dtype::TypeCommon;
    use hpt_types::vectors::traits::VecTrait;

    /// A SIMD-optimized mutable strided iterator over tensor elements.
    ///
    /// This struct provides mutable access to tensor elements with SIMD optimizations.
    pub struct StridedMutSimd<'a, T: TypeCommon> {
        /// The underlying SIMD-optimized strided iterator.
        pub(crate) base: StridedSimd<T>,
        /// The stride for the last dimension, used for inner loop element access.
        pub(crate) last_stride: i64,
        /// Phantom data to associate the lifetime `'a` with the struct.
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, T: CommonBounds> StridedMutSimd<'a, T> {
        /// Creates a new `StridedMutSimd` instance from a tensor.
        ///
        /// This constructor initializes a `StridedMutSimd` iterator by creating a base `StridedSimd`
        /// from the provided tensor. It also retrieves the last stride from the base iterator to
        /// configure the strided access pattern. The `PhantomData` marker is used to associate
        /// the iterator with the tensor's data type `T` without holding any actual data.
        ///
        /// # Arguments
        ///
        /// * `tensor` - An instance that implements the `TensorInfo<T>` trait, representing the tensor
        ///   to be iterated over. This tensor provides the necessary information about the tensor's shape,
        ///   strides, and data layout.
        ///
        /// # Returns
        ///
        /// A new instance of `StridedMutSimd`
        pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
            let base = StridedSimd::new(tensor);
            let last_stride = base.last_stride;
            StridedMutSimd {
                base,
                last_stride,
                phantom: std::marker::PhantomData,
            }
        }
    }

    impl<'a, T: 'a> IterGetSetSimd for StridedMutSimd<'a, T>
    where
        T: CommonBounds,
    {
        type Item = &'a mut T;
        type SimdItem = &'a mut T::Vec;

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
            todo!()
        }
        #[inline(always)]
        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            unsafe {
                &mut *self
                    .base
                    .ptr
                    .ptr
                    .offset((index as isize) * (self.last_stride as isize))
            }
        }
        fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem {
            let vector = unsafe { self.base.ptr.ptr.add(index * T::Vec::SIZE) };
            unsafe { std::mem::transmute(vector) }
        }
        fn all_last_stride_one(&self) -> bool {
            self.base.all_last_stride_one()
        }

        fn lanes(&self) -> Option<usize> {
            self.base.lanes()
        }
    }
    impl<'a, T> StridedIteratorSimd for StridedMutSimd<'a, T> where T: CommonBounds {}
    impl<'a, T> StridedSimdIteratorZip for StridedMutSimd<'a, T> where T: CommonBounds {}
}

/// A mutable strided iterator over tensor elements.
///
/// This struct provides mutable access to tensor elements with strided access patterns in `single thread`.
pub struct StridedMut<'a, T> {
    /// The underlying `single thread` strided iterator handling the iteration logic.
    pub(crate) base: Strided<T>,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: CommonBounds> StridedMut<'a, T> {
    /// Creates a new `StridedMut` instance from a given tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over.
    ///
    /// # Returns
    ///
    /// A new instance of `StridedMut` initialized with the provided tensor.
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        StridedMut {
            base: Strided::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }
    /// Combines this `StridedMut` iterator with another iterator, enabling simultaneous iteration.
    ///
    /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
    /// iterate over tensors with compatible shapes. It adjusts the strides and shapes of both iterators
    /// to match the broadcasted shape and then returns a `StridedZip` that allows for synchronized
    /// iteration over both iterators.
    ///
    /// # Arguments
    ///
    /// * `other` - The other iterator to zip with. It must implement the `IterGetSet` trait, and
    ///             its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A `StridedZip` instance that zips together `self` and `other`, enabling synchronized
    /// iteration over their elements.
    ///
    /// # Panics
    ///
    /// This method will panic if the shapes of `self` and `other` cannot be broadcasted together.
    /// Ensure that the shapes are compatible before calling this method.
    #[track_caller]
    pub fn zip<C>(mut self, mut other: C) -> StridedZip<'a, Self, C>
    where
        C: 'a + IterGetSet,
        <C as IterGetSet>::Item: Send,
    {
        let new_shape = match predict_broadcast_shape(self.shape(), other.shape()) {
            Ok(s) => s,
            Err(err) => {
                panic!("{}", err);
            }
        };

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        StridedZip::new(self, other)
    }
}

impl<'a, T: CommonBounds> StridedIterator for StridedMut<'a, T> {}
impl<'a, T: CommonBounds> StridedIteratorZip for StridedMut<'a, T> {}

impl<'a, T: 'a> IterGetSet for StridedMut<'a, T>
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
