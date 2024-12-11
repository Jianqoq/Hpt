use crate::{
    iterator_traits::{
        IterGetSet,
        ShapeManipulator,
        StridedHelper,
        StridedIterator,
        StridedIteratorMap,
        StridedIteratorZip,
    },
    shape_manipulate::{ expand, reshape, transpose },
};
use std::sync::Arc;
use tensor_common::{
    axis::Axis,
    layout::Layout,
    pointer::Pointer,
    shape::Shape,
    shape_utils::try_pad_shape,
    strides::Strides,
    strides_utils::preprocess_strides,
};
use tensor_traits::tensor::{ CommonBounds, TensorInfo };

/// A module for single-threaded strided simd iterator.
pub mod strided_simd {
    use std::sync::Arc;
    use tensor_common::{
        axis::Axis,
        layout::Layout,
        pointer::Pointer,
        shape::Shape,
        shape_utils::try_pad_shape,
        strides::Strides,
        strides_utils::preprocess_strides,
    };
    use tensor_traits::{ CommonBounds, TensorInfo };
    use tensor_types::dtype::TypeCommon;
    use tensor_types::vectors::traits::VecTrait;

    use crate::iterator_traits::{
        IterGetSetSimd,
        ShapeManipulator,
        StridedIteratorMap,
        StridedIteratorSimd,
        StridedSimdIteratorZip,
    };

    use super::{ expand, reshape, transpose, StridedHelper };

    /// A single thread SIMD-optimized strided iterator
    #[derive(Clone)]
    pub struct StridedSimd<T: TypeCommon> {
        /// A pointer to the tensor's data.
        pub(crate) ptr: Pointer<T>,
        /// The layout of the tensor, including shape and strides.
        pub(crate) layout: Layout,
        /// The loop progress of the iterator.
        pub(crate) prg: Vec<i64>,
        /// The stride for the last dimension.
        pub(crate) last_stride: i64,
    }

    impl<T: CommonBounds> StridedSimd<T> {
        /// Retrieves the shape of the tensor.
        ///
        /// # Returns
        ///
        /// A reference to the `Shape` struct representing the tensor's dimensions.
        pub fn shape(&self) -> &Shape {
            self.layout.shape()
        }
        /// Retrieves the strides of the tensor.
        ///
        /// # Returns
        ///
        /// A reference to the `Strides` struct representing the tensor's stride information.
        pub fn strides(&self) -> &Strides {
            self.layout.strides()
        }
        /// Creates a new `StridedSimd` instance from a given tensor.
        ///
        /// # Arguments
        ///
        /// * `tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
        ///
        /// # Returns
        ///
        /// A new instance of `StridedSimd` initialized with the provided tensor.
        pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
            StridedSimd {
                ptr: tensor.ptr(),
                layout: tensor.layout().clone(),
                prg: vec![],
                last_stride: *tensor.strides().last().unwrap_or(&0),
            }
        }
    }

    impl<T: CommonBounds> IterGetSetSimd for StridedSimd<T> {
        type Item = T;
        type SimdItem = T::Vec;

        fn set_end_index(&mut self, _: usize) {
            panic!("single thread iterator does not support set_end_index");
        }

        fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {
            panic!("single thread iterator does not support set_intervals");
        }

        fn set_strides(&mut self, strides: Strides) {
            self.layout.set_strides(strides);
        }

        fn set_shape(&mut self, shape: Shape) {
            self.layout.set_shape(shape);
        }

        fn set_prg(&mut self, prg: Vec<i64>) {
            self.prg = prg;
        }

        fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
            panic!("single thread iterator does not support intervals");
        }

        fn strides(&self) -> &Strides {
            self.layout.strides()
        }

        fn shape(&self) -> &Shape {
            self.layout.shape()
        }

        fn layout(&self) -> &Layout {
            &self.layout
        }

        fn broadcast_set_strides(&mut self, shape: &Shape) {
            let self_shape = try_pad_shape(self.shape(), shape.len());
            self.set_strides(preprocess_strides(&self_shape, self.strides()).into());
            self.last_stride = self.strides()[self.strides().len() - 1];
        }

        fn outer_loop_size(&self) -> usize {
            (self.shape().iter().product::<i64>() as usize) / self.inner_loop_size()
        }
        fn inner_loop_size(&self) -> usize {
            self.shape().last().unwrap().clone() as usize
        }

        fn next(&mut self) {
            for j in (0..(self.shape().len() as i64) - 1).rev() {
                let j = j as usize;
                if self.prg[j] < self.shape()[j] - 1 {
                    self.prg[j] += 1;
                    self.ptr.offset(self.strides()[j]);
                    break;
                } else {
                    self.prg[j] = 0;
                    self.ptr.offset(-self.strides()[j] * (self.shape()[j] - 1));
                }
            }
        }
        fn next_simd(&mut self) {
            todo!()
        }
        #[inline(always)]
        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            unsafe { *self.ptr.ptr.offset((index as isize) * (self.last_stride as isize)) }
        }
        fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem {
            unsafe { Self::SimdItem::from_ptr(self.ptr.get_ptr().add(index * T::Vec::SIZE)) }
        }
        fn all_last_stride_one(&self) -> bool {
            self.last_stride == 1
        }

        fn lanes(&self) -> Option<usize> {
            Some(T::Vec::SIZE)
        }
    }

    impl<T: CommonBounds> ShapeManipulator for StridedSimd<T> {
        #[track_caller]
        fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
            reshape(self, shape)
        }

        fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self {
            transpose(self, axes)
        }

        fn expand<S: Into<Shape>>(self, shape: S) -> Self {
            expand(self, shape)
        }
    }

    impl<T: CommonBounds> StridedHelper for StridedSimd<T> {
        fn _set_last_strides(&mut self, stride: i64) {
            self.last_stride = stride;
        }
        fn _set_strides(&mut self, strides: Strides) {
            self.layout.set_strides(strides);
        }
        fn _set_shape(&mut self, shape: Shape) {
            self.layout.set_shape(shape);
        }
        fn _layout(&self) -> &Layout {
            &self.layout
        }
    }
    impl<T: CommonBounds> StridedIteratorMap for StridedSimd<T> {}
    impl<T: CommonBounds> StridedSimdIteratorZip for StridedSimd<T> {}
    impl<T> StridedIteratorSimd for StridedSimd<T> where T: CommonBounds {}
}

/// A single-threaded strided iterator over tensor elements.
#[derive(Clone)]
pub struct Strided<T> {
    /// A pointer points to the tensor's data.
    pub(crate) ptr: Pointer<T>,
    /// The layout of the tensor, including shape and strides.
    pub(crate) layout: Layout,
    /// The loop progress of the iterator.
    pub(crate) prg: Vec<i64>,
    /// The stride for the last dimension.
    pub(crate) last_stride: i64,
}

impl<T: CommonBounds> Strided<T> {
    /// Retrieves the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A reference to the `Shape` struct representing the tensor's dimensions.
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }
    /// Retrieves the strides of the tensor.
    ///
    /// # Returns
    ///
    /// A reference to the `Strides` struct representing the tensor's stride information.
    pub fn strides(&self) -> &Strides {
        self.layout.strides()
    }
    /// Creates a new `Strided` instance from a given tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
    ///
    /// # Returns
    ///
    /// A new instance of `Strided` initialized with the provided tensor.
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        Strided {
            ptr: tensor.ptr(),
            layout: tensor.layout().clone(),
            prg: vec![],
            last_stride: *tensor.strides().last().unwrap_or(&0),
        }
    }
}

impl<T> StridedIteratorMap for Strided<T> {}
impl<T> StridedIteratorZip for Strided<T> {}

impl<T: CommonBounds> IterGetSet for Strided<T> {
    type Item = T;

    fn set_end_index(&mut self, _: usize) {
        panic!("single thread iterator does not support set_end_index");
    }

    fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {
        panic!("single thread iterator does not support set_intervals");
    }

    fn set_strides(&mut self, strides: Strides) {
        self.layout.set_strides(strides);
    }

    fn set_shape(&mut self, shape: Shape) {
        self.layout.set_shape(shape);
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.prg = prg;
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        panic!("single thread iterator does not support intervals");
    }
    fn strides(&self) -> &Strides {
        self.layout.strides()
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        let self_shape = try_pad_shape(self.shape(), shape.len());
        self.set_strides(preprocess_strides(&self_shape, self.strides()).into());
        self.last_stride = self.strides()[self.strides().len() - 1];
    }

    fn outer_loop_size(&self) -> usize {
        (self.shape().iter().product::<i64>() as usize) / self.inner_loop_size()
    }

    fn inner_loop_size(&self) -> usize {
        self.shape().last().unwrap().clone() as usize
    }

    fn next(&mut self) {
        for j in (0..(self.shape().len() as i64) - 1).rev() {
            let j = j as usize;
            if self.prg[j] < self.shape()[j] - 1 {
                self.prg[j] += 1;
                self.ptr.offset(self.strides()[j]);
                break;
            } else {
                self.prg[j] = 0;
                self.ptr.offset(-self.strides()[j] * (self.shape()[j] - 1));
            }
        }
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        unsafe { *self.ptr.get_ptr().add(index * (self.last_stride as usize)) }
    }
}

impl<T: CommonBounds> ShapeManipulator for Strided<T> {
    #[track_caller]
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
        reshape(self, shape)
    }

    fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self {
        transpose(self, axes)
    }

    fn expand<S: Into<Shape>>(self, shape: S) -> Self {
        expand(self, shape)
    }
}

impl<T: CommonBounds> StridedIterator for Strided<T> {}

impl<T> StridedHelper for Strided<T> {
    fn _set_last_strides(&mut self, stride: i64) {
        self.last_stride = stride;
    }
    fn _set_strides(&mut self, strides: Strides) {
        self.layout.set_strides(strides);
    }
    fn _set_shape(&mut self, shape: Shape) {
        self.layout.set_shape(shape);
    }
    fn _layout(&self) -> &Layout {
        &self.layout
    }
}
