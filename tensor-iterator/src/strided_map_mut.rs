use std::sync::Arc;
use tensor_common::{ shape::Shape, strides::Strides };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };

use crate::{
    iterator_traits::{ IterGetSet, StridedIterator },
    par_strided_mut::ParStridedMut,
    strided_zip::StridedZip,
};

pub struct StridedMapMut<'a, T> where T: Copy {
    pub(crate) base: ParStridedMut<'a, T>,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> StridedMapMut<'a, T> where T: CommonBounds {
    pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
        StridedMapMut {
            base: ParStridedMut::new(res_tensor),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zip<C>(self, other: C) -> StridedZip<'a, Self, C>
        where C: 'a + IterGetSet, <C as IterGetSet>::Item: Send
    {
        StridedZip::new(self, other)
    }
}

impl<'a, T> StridedIterator for StridedMapMut<'a, T> where T: 'a + CommonBounds {
    type Item = &'a mut T;
    fn for_each<F>(self, _: F) where F: Fn(Self::Item) {
        unimplemented!()
    }
}

impl<'a, T: 'a + CommonBounds> IterGetSet for StridedMapMut<'a, T> {
    type Item = &'a mut T;

    fn set_end_index(&mut self, _: usize) {}

    fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {}

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
