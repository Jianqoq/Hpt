use crate::{iterator_traits::IterGetSet, par_strided::ParStrided, par_strided_zip::ParStridedZip};
use rayon::iter::{
    plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};
use std::sync::Arc;
use tensor_common::{
    shape::Shape,
    shape_utils::{mt_intervals, predict_broadcast_shape},
};
use tensor_traits::tensor::{CommonBounds, TensorInfo};

pub mod par_strided_map_mut_simd {
    use crate::{
        iterator_traits::IterGetSetSimd, par_strided::par_strided_simd::ParStridedSimd,
        par_strided_zip::par_strided_zip_simd::ParStridedZipSimd,
    };
    use rayon::iter::{
        plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
        ParallelIterator,
    };
    use std::sync::Arc;
    use tensor_common::{
        shape::Shape,
        shape_utils::{mt_intervals, predict_broadcast_shape},
    };
    use tensor_traits::{CommonBounds, TensorInfo};
    use tensor_types::dtype::TypeCommon;
    use tensor_types::vectors::traits::VecCommon;

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
        /// Combines this `ParStridedMutSimd` iterator with another SIMD-optimized iterator, enabling simultaneous parallel iteration.
        ///
        /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
        /// iterate over tensors with compatible shapes. It calculates the appropriate iteration intervals based
        /// on the new broadcasted shape and configures both iterators accordingly. Finally, it returns a new
        /// `ParStridedZipSimd` instance that allows for synchronized parallel iteration over the combined iterators.
        ///
        /// **Note:** This implementation leverages Rayon for parallel execution and assumes that the iterators
        /// support SIMD optimizations.
        ///
        /// # Arguments
        ///
        /// * `other` - The third iterator to zip with. It must implement the `IterGetSetSimd`, `UnindexedProducer`,
        ///             `ParallelIterator` traits, and its associated `Item` type must be `Send`.
        ///
        /// # Returns
        ///
        /// A new `ParStridedZipSimd` instance that combines `self` and `other` for synchronized parallel iteration over both iterators.
        ///
        /// # Panics
        ///
        /// This method will panic if the shapes of `self` and `other` cannot be broadcasted together.
        /// Ensure that the shapes are compatible before calling this method.
        #[track_caller]
        pub fn zip<C>(mut self, mut other: C) -> ParStridedZipSimd<'a, Self, C>
        where
            C: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator,
            <C as IterGetSetSimd>::Item: Send,
            T::Vec: Send,
        {
            let new_shape = predict_broadcast_shape(self.shape(), other.shape())
                .expect("Cannot broadcast shapes");

            let inner_loop_size = new_shape[new_shape.len() - 1] as usize;

            // if collapse all is true, then the outer loop size is the product of all the elements in the shape
            // inner_loop_size in this case will be useless
            let outer_loop_size = (new_shape.size() as usize) / inner_loop_size;
            let num_threads;
            if outer_loop_size < rayon::current_num_threads() {
                num_threads = outer_loop_size;
            } else {
                num_threads = rayon::current_num_threads();
            }
            let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
            let len = intervals.len();
            self.set_intervals(intervals.clone());
            self.set_end_index(len);
            other.set_intervals(intervals.clone());
            other.set_end_index(len);

            other.broadcast_set_strides(&new_shape);
            self.broadcast_set_strides(&new_shape);

            other.set_shape(new_shape.clone());
            self.set_shape(new_shape.clone());

            ParStridedZipSimd::new(self, other)
        }
    }

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

        type SimdItem = &'a mut T::Vec where Self: 'a;

        fn set_end_index(&mut self, end_index: usize) {
            self.base.set_end_index(end_index);
        }

        fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.base.set_intervals(intervals);
        }

        fn set_strides(&mut self, strides: tensor_common::strides::Strides) {
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

        fn strides(&self) -> &tensor_common::strides::Strides {
            self.base.strides()
        }

        fn shape(&self) -> &Shape {
            self.base.shape()
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

        fn inner_loop_next_simd(&self, index: usize) -> Self::SimdItem {
            unsafe {
                std::mem::transmute(
                    self.base
                        .ptr
                        .get_ptr()
                        .add(index * T::Vec::SIZE)
                        .as_mut()
                        .unwrap(),
                )
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
    /// Combines this `ParStridedMut` iterator with another iterator, enabling simultaneous parallel iteration.
    ///
    /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
    /// iterate over tensors with compatible shapes. It calculates the appropriate iteration intervals based
    /// on the new broadcasted shape and configures both iterators accordingly. Finally, it returns a new
    /// `ParStridedZip` instance that allows for synchronized parallel iteration over the combined iterators.
    ///
    /// **Note:** This implementation leverages Rayon for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `other` - The other iterator to zip with. It must implement the `IterGetSet`, `UnindexedProducer`,
    ///             and `ParallelIterator` traits, and its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A new `ParStridedZip` instance that combines `self` and `other` for synchronized parallel iteration over both iterators.
    #[track_caller]
    pub fn zip<C>(mut self, mut other: C) -> ParStridedZip<'a, Self, C>
    where
        C: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
        <C as IterGetSet>::Item: Send,
        T::Vec: Send,
    {
        let new_shape =
            predict_broadcast_shape(self.shape(), other.shape()).expect("Cannot broadcast shapes");

        let inner_loop_size = new_shape[new_shape.len() - 1] as usize;

        // if collapse all is true, then the outer loop size is the product of all the elements in the shape
        // inner_loop_size in this case will be useless
        let outer_loop_size = (new_shape.size() as usize) / inner_loop_size;
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
        let len = intervals.len();
        self.set_intervals(intervals.clone());
        self.set_end_index(len);
        other.set_intervals(intervals.clone());
        other.set_end_index(len);

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        ParStridedZip::new(self, other)
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

    fn set_strides(&mut self, strides: tensor_common::strides::Strides) {
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

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.base.strides()
    }

    fn shape(&self) -> &Shape {
        self.base.shape()
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
