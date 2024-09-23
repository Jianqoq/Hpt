use crate::{
    iterator_traits::{IterGetSet, ParStridedHelper, ParStridedIteratorZip, ShapeManipulator},
    par_strided_fold::ParStridedFold,
    par_strided_map::ParStridedMap,
    shape_manipulate::{par_expand, par_reshape, par_transpose},
};
use rayon::iter::{
    plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};
use std::sync::Arc;
use tensor_common::{
    axis::Axis,
    layout::Layout,
    pointer::Pointer,
    shape::Shape,
    shape_utils::{mt_intervals, try_pad_shape},
    strides::Strides,
    strides_utils::preprocess_strides,
};
use tensor_traits::tensor::{CommonBounds, TensorInfo};

/// A module for parallel strided iterators.
pub mod par_strided_simd {
    use std::sync::Arc;

    use rayon::iter::{
        plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
        ParallelIterator,
    };
    use tensor_common::{
        axis::Axis,
        layout::Layout,
        pointer::Pointer,
        shape::Shape,
        shape_utils::{mt_intervals, try_pad_shape},
        simd_ref::MutVec,
        strides::Strides,
        strides_utils::preprocess_strides,
    };
    use tensor_traits::{CommonBounds, TensorInfo};

    use crate::{
        iterator_traits::{
            IterGetSetSimd, ParStridedHelper, ParStridedIteratorSimd, ParStridedIteratorSimdZip, ShapeManipulator
        },
        par_strided_map::par_strided_map_simd::ParStridedMapSimd,
        shape_manipulate::{par_expand, par_reshape, par_transpose},
    };

    /// Parallel strided iterator for SIMD operations.
    ///
    /// This struct is used to iterate over a tensor in parallel, with SIMD support.
    #[derive(Clone)]
    pub struct ParStridedSimd<T: Send + Copy + Sync> {
        /// Pointer to the data.
        pub(crate) ptr: Pointer<T>,
        /// Layout of the tensor.
        pub(crate) layout: Layout,
        /// Progress of the loop.
        pub(crate) prg: Vec<i64>,
        /// Chunk intervals for the outer loop.
        pub(crate) intervals: Arc<Vec<(usize, usize)>>,
        /// Start index of the chunk intervals.
        pub(crate) start_index: usize,
        /// End index of the chunk intervals.
        pub(crate) end_index: usize,
        /// Stride of the last dimension.
        pub(crate) last_stride: i64,
    }
    impl<T: CommonBounds> ParStridedSimd<T> {
        /// Returns the shape of the iterator.
        pub fn shape(&self) -> &Shape {
            self.layout.shape()
        }

        /// Returns the strides of the iterator.
        pub fn strides(&self) -> &Strides {
            self.layout.strides()
        }

        /// Create a new parallel strided iterator for SIMD operations.
        pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
            let inner_loop_size = tensor.shape()[tensor.shape().len() - 1] as usize;
            let outer_loop_size = tensor.size() / inner_loop_size;
            let num_threads;
            if outer_loop_size < rayon::current_num_threads() {
                num_threads = outer_loop_size;
            } else {
                num_threads = rayon::current_num_threads();
            }
            let intervals = mt_intervals(outer_loop_size, num_threads);
            let len = intervals.len();
            ParStridedSimd {
                ptr: tensor.ptr(),
                layout: tensor.layout().clone(),
                prg: vec![],
                intervals: Arc::new(intervals),
                start_index: 0,
                end_index: len,
                last_stride: tensor.strides()[tensor.strides().len() - 1],
            }
        }

        /// Map the iterator with a function.
        pub fn strided_map_simd<'a, F, F2>(
            self,
            f: F,
            vec_op: F2,
        ) -> ParStridedMapSimd<'a, ParStridedSimd<T>, T, F, F2>
        where
            F: Fn((&mut T, <Self as IterGetSetSimd>::Item)) + Sync + Send + 'a,
            <Self as IterGetSetSimd>::Item: Send,
            F2: Send + Sync + Copy + Fn((MutVec<'_, T::Vec>, <Self as IterGetSetSimd>::SimdItem)),
        {
            {
                ParStridedMapSimd {
                    iter: self,
                    f,
                    f2: vec_op,
                    phantom: std::marker::PhantomData,
                }
            }
        }
    }

    impl<T: CommonBounds> ParStridedIteratorSimdZip for ParStridedSimd<T> {}
    impl<T: CommonBounds> ParStridedIteratorSimd for ParStridedSimd<T> {}

    impl<T: CommonBounds> IterGetSetSimd for ParStridedSimd<T>
    where
        T::Vec: Send,
    {
        type Item = T;

        type SimdItem = T::Vec;

        fn set_end_index(&mut self, end_index: usize) {
            self.end_index = end_index;
        }

        fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.intervals = intervals;
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
            &self.intervals
        }

        fn strides(&self) -> &Strides {
            self.layout.strides()
        }

        fn shape(&self) -> &Shape {
            self.layout.shape()
        }

        fn broadcast_set_strides(&mut self, shape: &Shape) {
            let self_shape = try_pad_shape(self.shape(), shape.len());
            self.set_strides(preprocess_strides(&self_shape, self.strides()).into());
            self.last_stride = self.strides()[self.strides().len() - 1];
        }

        fn outer_loop_size(&self) -> usize {
            self.intervals[self.start_index].1 - self.intervals[self.start_index].0
        }

        fn inner_loop_size(&self) -> usize {
            self.shape().last().unwrap().clone() as usize
        }

        fn next(&mut self) {
            for j in (0..(self.shape().len() as i64) - 1).rev() {
                let j = j as usize;
                if self.prg[j] < self.shape()[j] {
                    self.prg[j] += 1;
                    self.ptr.offset(self.strides()[j]);
                    break;
                } else {
                    self.prg[j] = 0;
                    self.ptr.offset(-self.strides()[j] * self.shape()[j]);
                }
            }
        }

        fn next_simd(&mut self) {}

        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            unsafe { *self.ptr.get_ptr().add(index * (self.last_stride as usize)) }
        }

        fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem {
            use tensor_types::vectors::traits::Init;
            use tensor_types::vectors::traits::VecCommon;
            unsafe { T::Vec::from_ptr(self.ptr.get_ptr().add(index * T::Vec::SIZE)) }
        }

        fn all_last_stride_one(&self) -> bool {
            self.last_stride == 1
        }

        fn lanes(&self) -> Option<usize> {
            use tensor_types::vectors::traits::VecCommon;
            Some(T::Vec::SIZE)
        }

        fn layout(&self) -> &Layout {
            &self.layout
        }
    }

    impl<T> ParallelIterator for ParStridedSimd<T>
    where
        T: CommonBounds,
        T::Vec: Send,
    {
        type Item = T;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge_unindexed(self, consumer)
        }
    }

    impl<T> UnindexedProducer for ParStridedSimd<T>
    where
        T: CommonBounds,
        T::Vec: Send,
    {
        type Item = T;

        fn split(mut self) -> (Self, Option<Self>) {
            if self.end_index - self.start_index <= 1 {
                let mut curent_shape_prg: Vec<i64> = vec![0; self.shape().len()];
                let mut amount =
                    self.intervals[self.start_index].0 * (*self.shape().last().unwrap() as usize);
                let mut index = 0;
                for j in (0..self.shape().len()).rev() {
                    curent_shape_prg[j] = (amount as i64) % self.shape()[j];
                    amount /= self.shape()[j] as usize;
                    index += curent_shape_prg[j] * self.strides()[j];
                }
                self.ptr.offset(index);
                self.prg = curent_shape_prg;
                let mut new_shape = self.shape().to_vec();
                new_shape.iter_mut().for_each(|x| {
                    *x -= 1;
                });
                self.last_stride = self.strides()[self.strides().len() - 1];
                self.set_shape(Shape::from(new_shape));
                return (self, None);
            }
            let _left_interval = &self.intervals[self.start_index..self.end_index];
            let left = _left_interval.len() / 2;
            let right = _left_interval.len() / 2 + (_left_interval.len() % 2);
            (
                ParStridedSimd {
                    ptr: self.ptr.clone(),
                    layout: self.layout.clone(),
                    prg: vec![],
                    intervals: self.intervals.clone(),
                    start_index: self.start_index,
                    end_index: self.start_index + left,
                    last_stride: self.last_stride,
                },
                Some(ParStridedSimd {
                    ptr: self.ptr.clone(),
                    layout: self.layout.clone(),
                    prg: vec![],
                    intervals: self.intervals.clone(),
                    start_index: self.start_index + left,
                    end_index: self.start_index + left + right,
                    last_stride: self.last_stride,
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

    impl<T: CommonBounds> ParStridedHelper for ParStridedSimd<T> {
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

        fn _set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.intervals = intervals;
        }

        fn _set_end_index(&mut self, end_index: usize) {
            self.end_index = end_index;
        }
    }

    impl<T: CommonBounds> ShapeManipulator for ParStridedSimd<T>
    where
        T::Vec: Send,
    {
        fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
            par_reshape(self, shape)
        }

        fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self {
            par_transpose(self, axes)
        }

        fn expand<S: Into<Shape>>(self, shape: S) -> Self {
            par_expand(self, shape)
        }
    }
}

/// A parallel strided iterator over tensor elements.
///
/// This struct provides mutable access to tensor elements with strided access patterns optimized for parallel processing.
#[derive(Clone)]
pub struct ParStrided<T> {
    /// A pointer to the tensor's data.
    pub(crate) ptr: Pointer<T>,
    /// The layout of the tensor, including shape and strides.
    pub(crate) layout: Layout,
    /// Progress of the loop.
    pub(crate) prg: Vec<i64>,
    /// Chunk intervals for the outer loop.
    pub(crate) intervals: Arc<Vec<(usize, usize)>>,
    /// Start index of the chunk intervals.
    pub(crate) start_index: usize,
    /// End index of the chunk intervals.
    pub(crate) end_index: usize,
    /// Stride of the last dimension.
    pub(crate) last_stride: i64,
}

impl<T: CommonBounds> ParStrided<T> {
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
    /// Creates a new `ParStrided` instance from a given tensor.
    ///
    /// This constructor initializes the `ParStrided` iterator by determining the appropriate number of threads
    /// based on the tensor's outer loop size and Rayon’s current number of threads. It then divides the
    /// iteration workload into intervals for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor implementing the `TensorInfo<T>` trait to iterate over mutably.
    ///
    /// # Returns
    ///
    /// A new instance of `ParStrided` initialized with the provided tensor.
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        let inner_loop_size = tensor.shape()[tensor.shape().len() - 1] as usize;
        let outer_loop_size = tensor.size() / inner_loop_size;
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = mt_intervals(outer_loop_size, num_threads);
        let len = intervals.len();
        ParStrided {
            ptr: tensor.ptr(),
            layout: tensor.layout().clone(),
            prg: vec![],
            intervals: Arc::new(intervals),
            start_index: 0,
            end_index: len,
            last_stride: tensor.strides()[tensor.strides().len() - 1],
        }
    }
    /// Performs a parallel fold (reduce) operation over the tensor elements.
    ///
    /// This method applies a folding function `fold_op` to accumulate tensor elements into an initial
    /// identity value `identity`. It leverages parallel iteration to perform the fold operation efficiently.
    ///
    /// # Type Parameters
    ///
    /// * `ID` - The type of the accumulator.
    /// * `F` - The folding function.
    ///
    /// # Arguments
    ///
    /// * `identity` - The initial value for the accumulator.
    /// * `fold_op` - A function that takes the current accumulator and an element, returning the updated accumulator.
    ///
    /// # Returns
    ///
    /// A `ParStridedFold` instance that represents the fold operation.
    pub fn par_strided_fold<ID, F>(self, identity: ID, fold_op: F) -> ParStridedFold<Self, ID, F>
    where
        F: Fn(ID, T) -> ID + Sync + Send + Copy,
        ID: Sync + Send + Copy,
    {
        ParStridedFold {
            iter: self,
            identity,
            fold_op,
        }
    }
    /// Transforms the zipped iterators by applying a provided function to their items.
    ///
    /// This method allows for element-wise operations on the zipped iterators by applying `func` to each item.
    ///
    /// # Type Parameters
    ///
    /// * `'a` - The lifetime associated with the iterators.
    /// * `F` - The function to apply to each item.
    /// * `U` - The output type after applying the function.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an item from the zipped iterator and returns a transformed value.
    ///
    /// # Returns
    ///
    /// A `ParStridedMap` instance that applies the provided function during iteration.
    pub fn strided_map<'a, F, U>(self, f: F) -> ParStridedMap<'a, ParStrided<T>, T, F>
    where
        F: Fn(T) -> U + Sync + Send + 'a,
        U: CommonBounds,
    {
        ParStridedMap {
            iter: self,
            f,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: CommonBounds> ParStridedIteratorZip for ParStrided<T> {}

impl<T: CommonBounds> IterGetSet for ParStrided<T> {
    type Item = T;

    fn set_end_index(&mut self, end_index: usize) {
        self.end_index = end_index;
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.intervals = intervals;
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
        &self.intervals
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        let self_shape = try_pad_shape(self.shape(), shape.len());
        self.set_strides(preprocess_strides(&self_shape, self.strides()).into());
        self.last_stride = self.strides()[self.strides().len() - 1];
    }

    fn next(&mut self) {
        for j in (0..(self.shape().len() as i64) - 1).rev() {
            let j = j as usize;
            if self.prg[j] < self.shape()[j] {
                self.prg[j] += 1;
                self.ptr.offset(self.strides()[j]);
                break;
            } else {
                self.prg[j] = 0;
                self.ptr.offset(-self.strides()[j] * self.shape()[j]);
            }
        }
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        unsafe { *self.ptr.get_ptr().add(index * (self.last_stride as usize)) }
    }

    fn strides(&self) -> &Strides {
        self.layout.strides()
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn outer_loop_size(&self) -> usize {
        self.intervals[self.start_index].1 - self.intervals[self.start_index].0
    }

    fn inner_loop_size(&self) -> usize {
        self.shape().last().unwrap().clone() as usize
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T> ParallelIterator for ParStrided<T>
where
    T: CommonBounds,
    T::Vec: Send,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }
}

impl<T> UnindexedProducer for ParStrided<T>
where
    T: CommonBounds,
    T::Vec: Send,
{
    type Item = T;

    fn split(mut self) -> (Self, Option<Self>) {
        if self.end_index - self.start_index <= 1 {
            let mut curent_shape_prg: Vec<i64> = vec![0; self.shape().len()];
            let mut amount =
                self.intervals[self.start_index].0 * (*self.shape().last().unwrap() as usize);
            let mut index = 0;
            for j in (0..self.shape().len()).rev() {
                curent_shape_prg[j] = (amount as i64) % self.shape()[j];
                amount /= self.shape()[j] as usize;
                index += curent_shape_prg[j] * self.strides()[j];
            }
            self.ptr.offset(index);
            self.prg = curent_shape_prg;
            let mut new_shape = self.shape().to_vec();
            new_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            self.last_stride = self.strides()[self.strides().len() - 1];
            self.set_shape(Shape::from(new_shape));
            return (self, None);
        }
        let _left_interval = &self.intervals[self.start_index..self.end_index];
        let left = _left_interval.len() / 2;
        let right = _left_interval.len() / 2 + (_left_interval.len() % 2);
        (
            ParStrided {
                ptr: self.ptr.clone(),
                layout: self.layout.clone(),
                prg: vec![],
                intervals: self.intervals.clone(),
                start_index: self.start_index,
                end_index: self.start_index + left,
                last_stride: self.last_stride,
            },
            Some(ParStrided {
                ptr: self.ptr.clone(),
                layout: self.layout.clone(),
                prg: vec![],
                intervals: self.intervals.clone(),
                start_index: self.start_index + left,
                end_index: self.start_index + left + right,
                last_stride: self.last_stride,
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

impl<T> ParStridedHelper for ParStrided<T> {
    fn _layout(&self) -> &Layout {
        &self.layout
    }

    fn _set_last_strides(&mut self, last_stride: i64) {
        self.last_stride = last_stride;
    }

    fn _set_strides(&mut self, strides: Strides) {
        self.layout.set_strides(strides);
    }

    fn _set_shape(&mut self, shape: Shape) {
        self.layout.set_shape(shape);
    }

    fn _set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.intervals = intervals;
    }

    fn _set_end_index(&mut self, end_index: usize) {
        self.end_index = end_index;
    }
}

impl<T: CommonBounds> ShapeManipulator for ParStrided<T>
where
    T::Vec: Send,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
        par_reshape(self, shape)
    }

    fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self {
        par_transpose(self, axes)
    }

    fn expand<S: Into<Shape>>(self, shape: S) -> Self {
        par_expand(self, shape)
    }
}