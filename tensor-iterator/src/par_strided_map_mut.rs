use std::sync::Arc;

use rayon::iter::{
    plumbing::{ bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer },
    ParallelIterator,
};
use tensor_common::{ shape::Shape, strides::Strides };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };

use crate::{ iterator_traits::IterGetSet, par_strided_mut::ParStridedMut, par_strided_zip::ParStridedZip };

pub struct ParStridedMapMut<'a, T> where T: Copy {
    pub(crate) base: ParStridedMut<'a, T>,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> ParStridedMapMut<'a, T> where T: CommonBounds {
    pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
        ParStridedMapMut {
            base: ParStridedMut::new(res_tensor),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zip<C>(self, other: C) -> ParStridedZip<'a, Self, C>
        where
            C: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
            <C as IterGetSet>::Item: Send
    {
        ParStridedZip::new(self, other)
    }
}

impl<'a, T> ParallelIterator for ParStridedMapMut<'a, T> where T: 'a + CommonBounds {
    type Item = &'a mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where C: UnindexedConsumer<Self::Item> {
        bridge_unindexed(self, consumer)
    }
}

impl<'a, T> UnindexedProducer for ParStridedMapMut<'a, T> where T: 'a + CommonBounds {
    type Item = &'a mut T;

    fn split(self) -> (Self, Option<Self>) {
        let (a, b) = self.base.split();
        (
            ParStridedMapMut { base: a, phantom: std::marker::PhantomData },
            b.map(|x| ParStridedMapMut { base: x, phantom: std::marker::PhantomData }),
        )
    }

    fn fold_with<F>(self, folder: F) -> F where F: Folder<Self::Item> {
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

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        self.base.intervals()
    }

    fn strides(&self) -> &Strides {
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
        self.base.inner_loop_next(index)
    }
    
    fn set_prg(&mut self, prg: Vec<i64>) {
        self.base.set_prg(prg);
    }
}
