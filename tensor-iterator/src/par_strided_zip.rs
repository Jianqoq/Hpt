use std::{panic::Location, sync::Arc};

use rayon::iter::{
    plumbing::{ bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer },
    ParallelIterator,
};
use tensor_common::{
    shape::Shape,
    shape_utils::{ mt_intervals, predict_broadcast_shape },
    strides::Strides,
};
use tensor_traits::tensor::CommonBounds;

use crate::iterator_traits::{ IterGetSet, ShapeManipulator };
use crate::par_strided_map::ParStridedMap;

#[derive(Clone)]
pub struct ParStridedZip<'a, A: 'a, B: 'a> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, A, B> IterGetSet for ParStridedZip<'a, A, B> where A: IterGetSet, B: IterGetSet {
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

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        self.a.intervals()
    }

    fn strides(&self) -> &Strides {
        self.a.strides()
    }

    fn shape(&self) -> &Shape {
        self.a.shape()
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

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.a.set_prg(prg.clone());
        self.b.set_prg(prg);
    }
}

impl<'a, A, B> ParStridedZip<'a, A, B>
    where
        A: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
        B: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
        <A as IterGetSet>::Item: Send,
        <B as IterGetSet>::Item: Send
{
    pub fn new(a: A, b: B) -> Self {
        ParStridedZip {
            a,
            b,
            phantom: std::marker::PhantomData,
        }
    }

    #[track_caller]
    pub fn zip<C>(mut self, mut other: C) -> ParStridedZip<'a, Self, C>
        where
            C: UnindexedProducer + IterGetSet + ShapeManipulator + ParallelIterator + IterGetSet,
            Self: UnindexedProducer + IterGetSet + ShapeManipulator,
            <C as IterGetSet>::Item: Send,
            <Self as IterGetSet>::Item: Send
    {
        let new_shape = predict_broadcast_shape(&self.shape(), &other.shape(), Location::caller()).expect(
            "Cannot broadcast shapes"
        );

        let inner_loop_size = new_shape[new_shape.len() - 1] as usize;
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

        let mut a = self.reshape(new_shape.clone());
        let mut b = other.reshape(new_shape.clone());

        a.set_shape(new_shape.clone());
        b.set_shape(new_shape.clone());
        ParStridedZip::new(a, b)
    }

    pub fn strided_map<F, U>(
        self,
        func: F
    )
        -> ParStridedMap<'a, Self, <Self as IterGetSet>::Item, F>
        where
            F: Fn(<Self as IterGetSet>::Item) -> U + Sync + Send + 'a,
            U: CommonBounds,
            <A as IterGetSet>::Item: Send,
            <B as IterGetSet>::Item: Send
    {
        ParStridedMap {
            iter: self,
            f: func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, A, B> UnindexedProducer
    for ParStridedZip<'a, A, B>
    where
        A: UnindexedProducer + ParallelIterator + IterGetSet,
        B: UnindexedProducer + ParallelIterator + IterGetSet
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

    fn fold_with<F>(mut self, mut folder: F) -> F where F: Folder<Self::Item> {
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

impl<'a, A, B> ParallelIterator
    for ParStridedZip<'a, A, B>
    where
        A: UnindexedProducer + ParallelIterator + IterGetSet,
        B: UnindexedProducer + ParallelIterator + IterGetSet,
        <A as IterGetSet>::Item: Send,
        <B as IterGetSet>::Item: Send
{
    type Item = (<A as IterGetSet>::Item, <B as IterGetSet>::Item);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where C: UnindexedConsumer<Self::Item> {
        bridge_unindexed(self, consumer)
    }
}
