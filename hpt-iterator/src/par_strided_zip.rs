use par_strided_zip_simd::ParStridedZipSimd;
use rayon::iter::{
    plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};
use std::sync::Arc;
use hpt_common::{shape::shape::Shape, strides::strides::Strides};
use hpt_traits::tensor::CommonBounds;

use crate::{
    iterator_traits::{IterGetSet, IterGetSetSimd, ParStridedIteratorZip, ShapeManipulator},
    par_strided_map::ParStridedMap,
};

/// A module for parallel strided simd zip iterator.
pub mod par_strided_zip_simd {
    use std::sync::Arc;

    use rayon::iter::{
        plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
        ParallelIterator,
    };
    use hpt_common::{shape::shape::Shape, strides::strides::Strides, utils::simd_ref::MutVec};
    use hpt_traits::CommonBounds;

    use crate::{
        iterator_traits::{
            IterGetSetSimd, ParStridedIteratorSimd, ParStridedIteratorSimdZip, ShapeManipulator,
        },
        par_strided_map::par_strided_map_simd::ParStridedMapSimd,
    };

    // A parallel SIMD-optimized zipped iterator combining two iterators over tensor elements.
    ///
    /// This struct allows for synchronized parallel iteration over two SIMD-optimized iterators,
    #[derive(Clone)]
    pub struct ParStridedZipSimd<'a, A: 'a, B: 'a> {
        /// The first iterator to be zipped.
        pub(crate) a: A,
        /// The second iterator to be zipped.
        pub(crate) b: B,
        /// Phantom data to associate the lifetime `'a` with the struct.
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, A, B> IterGetSetSimd for ParStridedZipSimd<'a, A, B>
    where
        A: IterGetSetSimd,
        B: IterGetSetSimd,
    {
        type Item = (<A as IterGetSetSimd>::Item, <B as IterGetSetSimd>::Item);

        type SimdItem = (
            <A as IterGetSetSimd>::SimdItem,
            <B as IterGetSetSimd>::SimdItem,
        );

        fn set_end_index(&mut self, end_index: usize) {
            self.a.set_end_index(end_index);
            self.b.set_end_index(end_index);
        }

        fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.a.set_intervals(intervals.clone());
            self.b.set_intervals(intervals);
        }

        fn set_strides(&mut self, last_stride: Strides) {
            self.a.set_strides(last_stride.clone());
            self.b.set_strides(last_stride);
        }

        fn set_shape(&mut self, shape: Shape) {
            self.a.set_shape(shape.clone());
            self.b.set_shape(shape);
        }

        fn set_prg(&mut self, prg: Vec<i64>) {
            self.a.set_prg(prg.clone());
            self.b.set_prg(prg);
        }

        fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
            self.a.intervals()
        }

        fn strides(&self) -> &Strides {
            self.a.strides()
        }

        fn shape(&self) -> &Shape {
            self.a.shape()
        }

        fn layout(&self) -> &hpt_common::layout::layout::Layout {
            self.a.layout()
        }

        fn broadcast_set_strides(&mut self, shape: &Shape) {
            self.a.broadcast_set_strides(shape);
            self.b.broadcast_set_strides(shape);
        }

        fn outer_loop_size(&self) -> usize {
            self.a.outer_loop_size()
        }

        fn inner_loop_size(&self) -> usize {
            self.a.inner_loop_size()
        }

        fn next(&mut self) {
            self.a.next();
            self.b.next();
        }

        fn next_simd(&mut self) {
            self.a.next_simd();
            self.b.next_simd();
        }

        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            (self.a.inner_loop_next(index), self.b.inner_loop_next(index))
        }

        fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem {
            (
                self.a.inner_loop_next_simd(index),
                self.b.inner_loop_next_simd(index),
            )
        }

        fn all_last_stride_one(&self) -> bool {
            self.a.all_last_stride_one() && self.b.all_last_stride_one()
        }

        fn lanes(&self) -> Option<usize> {
            match (self.a.lanes(), self.b.lanes()) {
                (Some(a), Some(b)) => {
                    if a == b {
                        Some(a)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
    }
    impl<'a, A, B> ParStridedZipSimd<'a, A, B>
    where
        A: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator,
        B: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator,
        <A as IterGetSetSimd>::Item: Send,
        <B as IterGetSetSimd>::Item: Send,
    {
        /// Creates a new `ParStridedZipSimd` instance by zipping two SIMD-optimized iterators.
        ///
        /// # Arguments
        ///
        /// * `a` - The first iterator to zip.
        /// * `b` - The second iterator to zip.
        ///
        /// # Returns
        ///
        /// A new `ParStridedZipSimd` instance that combines both iterators for synchronized parallel iteration.
        pub fn new(a: A, b: B) -> Self {
            ParStridedZipSimd {
                a,
                b,
                phantom: std::marker::PhantomData,
            }
        }
        /// Transforms the zipped iterators by applying provided functions to their items.
        ///
        /// This method allows for element-wise operations on the zipped iterators by applying `func` to each pair of items.
        ///
        /// # Arguments
        ///
        /// * `func` - A function that takes a mutable reference to a target and an item from the first iterator.
        /// * `func2` - A function that takes a mutable reference to a SIMD vector and an item from the second iterator.
        ///
        /// # Returns
        ///
        /// A `ParStridedMapSimd` instance that applies the provided functions during iteration.
        ///
        pub fn strided_map_simd<F, F2, T>(
            self,
            func: F,
            func2: F2,
        ) -> ParStridedMapSimd<'a, Self, <Self as IterGetSetSimd>::Item, F, F2>
        where
            F: Fn((&mut T, <Self as IterGetSetSimd>::Item)) + Sync + Send + 'a,
            F2: Fn((MutVec<'_, T::Vec>, <Self as IterGetSetSimd>::SimdItem)) + Sync + Send + 'a,
            T: CommonBounds,
            <A as IterGetSetSimd>::Item: Send,
            <B as IterGetSetSimd>::Item: Send,
            T::Vec: Send,
            A: ShapeManipulator,
            B: ShapeManipulator,
        {
            ParStridedMapSimd {
                iter: self,
                f: func,
                f2: func2,
                phantom: std::marker::PhantomData,
            }
        }
    }

    impl<'a, A, B> ParStridedIteratorSimdZip for ParStridedZipSimd<'a, A, B>
    where
        A: UnindexedProducer + ParallelIterator + IterGetSetSimd,
        B: UnindexedProducer + ParallelIterator + IterGetSetSimd,
    {
    }
    impl<'a, A, B> ParStridedIteratorSimd for ParStridedZipSimd<'a, A, B>
    where
        A: UnindexedProducer + ParallelIterator + IterGetSetSimd,
        B: UnindexedProducer + ParallelIterator + IterGetSetSimd,
        <A as IterGetSetSimd>::Item: Send,
        <B as IterGetSetSimd>::Item: Send,
    {
    }

    impl<'a, A, B> UnindexedProducer for ParStridedZipSimd<'a, A, B>
    where
        A: UnindexedProducer + ParallelIterator + IterGetSetSimd,
        B: UnindexedProducer + ParallelIterator + IterGetSetSimd,
    {
        type Item = <Self as IterGetSetSimd>::Item;

        fn split(self) -> (Self, Option<Self>) {
            let (left_a, right_a) = self.a.split();
            let (left_b, right_b) = self.b.split();
            if right_a.is_none() {
                (
                    ParStridedZipSimd {
                        a: left_a,
                        b: left_b,
                        phantom: std::marker::PhantomData,
                    },
                    None,
                )
            } else {
                (
                    ParStridedZipSimd {
                        a: left_a,
                        b: left_b,
                        phantom: std::marker::PhantomData,
                    },
                    Some(ParStridedZipSimd {
                        a: right_a.unwrap(),
                        b: right_b.unwrap(),
                        phantom: std::marker::PhantomData,
                    }),
                )
            }
        }

        fn fold_with<F>(mut self, mut folder: F) -> F
        where
            F: Folder<Self::Item>,
        {
            let outer_loop_size = self.outer_loop_size();
            let inner_loop_size = self.inner_loop_size() + 1;
            for _ in 0..outer_loop_size {
                for idx in 0..inner_loop_size {
                    folder = folder.consume(self.inner_loop_next(idx));
                }
                self.next();
            }
            folder
        }
    }

    impl<'a, A, B> ParallelIterator for ParStridedZipSimd<'a, A, B>
    where
        A: UnindexedProducer + ParallelIterator + IterGetSetSimd,
        B: UnindexedProducer + ParallelIterator + IterGetSetSimd,
        <A as IterGetSetSimd>::Item: Send,
        <B as IterGetSetSimd>::Item: Send,
    {
        type Item = (<A as IterGetSetSimd>::Item, <B as IterGetSetSimd>::Item);

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge_unindexed(self, consumer)
        }
    }
}

/// A parallel zipped iterator combining two iterators over tensor elements.
///
/// This struct allows for synchronized parallel iteration over two iterators
///
/// # Type Parameters
///
/// * `'a` - The lifetime associated with the iterators.
/// * `A` - The type of the first iterator.
/// * `B` - The type of the second iterator.
#[derive(Clone)]
pub struct ParStridedZip<'a, A: 'a, B: 'a> {
    /// The first iterator to be zipped.
    pub(crate) a: A,
    /// The second iterator to be zipped.
    pub(crate) b: B,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, A, B> IterGetSet for ParStridedZip<'a, A, B>
where
    A: IterGetSet,
    B: IterGetSet,
{
    type Item = (<A as IterGetSet>::Item, <B as IterGetSet>::Item);

    fn set_end_index(&mut self, end_index: usize) {
        self.a.set_end_index(end_index);
        self.b.set_end_index(end_index);
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.a.set_intervals(intervals.clone());
        self.b.set_intervals(intervals);
    }

    fn set_strides(&mut self, last_stride: Strides) {
        self.a.set_strides(last_stride.clone());
        self.b.set_strides(last_stride);
    }

    fn set_shape(&mut self, shape: Shape) {
        self.a.set_shape(shape.clone());
        self.b.set_shape(shape);
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.a.set_prg(prg.clone());
        self.b.set_prg(prg);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        self.a.intervals()
    }

    fn strides(&self) -> &Strides {
        self.a.strides()
    }

    fn shape(&self) -> &Shape {
        self.a.shape()
    }

    fn layout(&self) -> &hpt_common::layout::layout::Layout {
        self.a.layout()
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        self.a.broadcast_set_strides(shape);
        self.b.broadcast_set_strides(shape);
    }

    fn outer_loop_size(&self) -> usize {
        self.a.outer_loop_size()
    }

    fn inner_loop_size(&self) -> usize {
        self.a.inner_loop_size()
    }

    fn next(&mut self) {
        self.a.next();
        self.b.next();
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        (self.a.inner_loop_next(index), self.b.inner_loop_next(index))
    }
}

impl<'a, A, B> ParStridedZip<'a, A, B>
where
    A: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
    B: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
    <A as IterGetSet>::Item: Send,
    <B as IterGetSet>::Item: Send,
{
    /// Creates a new `ParStridedZip` instance by zipping two iterators.
    ///
    /// # Arguments
    ///
    /// * `a` - The first iterator to zip.
    /// * `b` - The second iterator to zip.
    ///
    /// # Returns
    ///
    /// A new `ParStridedZip` instance that combines both iterators for synchronized parallel iteration.
    pub fn new(a: A, b: B) -> Self {
        ParStridedZip {
            a,
            b,
            phantom: std::marker::PhantomData,
        }
    }
    /// Transforms the zipped iterators by applying a provided function to their items.
    ///
    /// This method allows for element-wise operations on the zipped iterators by applying `func` to each item.
    ///
    /// # Arguments
    ///
    /// * `func` - A function that takes an item from the zipped iterator and returns a transformed value.
    ///
    /// # Returns
    ///
    /// A `ParStridedMap` instance that applies the provided function during iteration.
    pub fn strided_map<F, T>(
        self,
        func: F,
    ) -> ParStridedMap<'a, Self, <Self as IterGetSet>::Item, F>
    where
        F: Fn((&mut T, <Self as IterGetSet>::Item)) + Sync + Send,
        T: CommonBounds,
    {
        ParStridedMap {
            iter: self,
            f: func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, A, B> UnindexedProducer for ParStridedZip<'a, A, B>
where
    A: UnindexedProducer + ParallelIterator + IterGetSet,
    B: UnindexedProducer + ParallelIterator + IterGetSet,
{
    type Item = <Self as IterGetSet>::Item;

    fn split(self) -> (Self, Option<Self>) {
        let (left_a, right_a) = self.a.split();
        let (left_b, right_b) = self.b.split();
        if right_a.is_none() {
            (
                ParStridedZip {
                    a: left_a,
                    b: left_b,
                    phantom: std::marker::PhantomData,
                },
                None,
            )
        } else {
            (
                ParStridedZip {
                    a: left_a,
                    b: left_b,
                    phantom: std::marker::PhantomData,
                },
                Some(ParStridedZip {
                    a: right_a.unwrap(),
                    b: right_b.unwrap(),
                    phantom: std::marker::PhantomData,
                }),
            )
        }
    }

    fn fold_with<F>(mut self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size() + 1;
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                folder = folder.consume(self.inner_loop_next(idx));
            }
            self.next();
        }
        folder
    }
}

impl<'a, A, B> ParallelIterator for ParStridedZip<'a, A, B>
where
    A: UnindexedProducer + ParallelIterator + IterGetSet,
    B: UnindexedProducer + ParallelIterator + IterGetSet,
    <A as IterGetSet>::Item: Send,
    <B as IterGetSet>::Item: Send,
{
    type Item = (<A as IterGetSet>::Item, <B as IterGetSet>::Item);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }
}

impl<'a, A, B> ParStridedIteratorZip for ParStridedZip<'a, A, B>
where
    A: UnindexedProducer + ParallelIterator + IterGetSet,
    B: UnindexedProducer + ParallelIterator + IterGetSet,
    <A as IterGetSet>::Item: Send,
    <B as IterGetSet>::Item: Send,
{
}

impl<'a, A, B> ShapeManipulator for ParStridedZip<'a, A, B>
where
    A: UnindexedProducer + 'a + IterGetSet + ParallelIterator + ShapeManipulator,
    B: UnindexedProducer + 'a + IterGetSet + ParallelIterator + ShapeManipulator,
    <A as IterGetSet>::Item: Send,
    <B as IterGetSet>::Item: Send,
{
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
        let tmp: Shape = shape.into();
        let a = self.a.reshape(tmp.clone());
        let b = self.b.reshape(tmp);
        ParStridedZip::new(a, b)
    }

    fn transpose<AXIS: Into<hpt_common::axis::axis::Axis>>(self, axes: AXIS) -> Self {
        let axes: hpt_common::axis::axis::Axis = axes.into();
        let a = self.a.transpose(axes.clone());
        let b = self.b.transpose(axes);
        ParStridedZip::new(a, b)
    }

    fn expand<S: Into<Shape>>(self, shape: S) -> Self {
        let tmp: Shape = shape.into();
        let a = self.a.expand(tmp.clone());
        let b = self.b.expand(tmp);
        ParStridedZip::new(a, b)
    }
}

impl<'a, A, B> ShapeManipulator for ParStridedZipSimd<'a, A, B>
where
    A: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator + ShapeManipulator,
    B: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator + ShapeManipulator,
    <A as IterGetSetSimd>::Item: Send,
    <B as IterGetSetSimd>::Item: Send,
{
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self {
        let tmp: Shape = shape.into();
        let a = self.a.reshape(tmp.clone());
        let b = self.b.reshape(tmp);
        ParStridedZipSimd::new(a, b)
    }

    fn transpose<AXIS: Into<hpt_common::axis::axis::Axis>>(self, axes: AXIS) -> Self {
        let axes: hpt_common::axis::axis::Axis = axes.into();
        let a = self.a.transpose(axes.clone());
        let b = self.b.transpose(axes);
        ParStridedZipSimd::new(a, b)
    }

    fn expand<S: Into<Shape>>(self, shape: S) -> Self {
        let tmp: Shape = shape.into();
        let a = self.a.expand(tmp.clone());
        let b = self.b.expand(tmp);
        ParStridedZipSimd::new(a, b)
    }
}
