use std::{panic::Location, sync::Arc};
use tensor_common::{ shape::Shape, shape_utils::predict_broadcast_shape };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use crate::{
    iterator_traits::{ IterGetSet, StridedIterator },
    strided::Strided,
    strided_zip::StridedZip,
};

pub struct StridedMut<'a, T> {
    pub(crate) base: Strided<T>,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: CommonBounds> StridedMut<'a, T> {
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        StridedMut {
            base: Strided::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }

    #[track_caller]
    pub fn zip<C>(mut self, mut other: C) -> StridedZip<'a, Self, C>
        where C: 'a + IterGetSet, <C as IterGetSet>::Item: Send
    {
        let new_shape = predict_broadcast_shape(self.shape(), other.shape(), Location::caller()).expect(
            "Cannot broadcast shapes"
        );

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        StridedZip::new(self, other)
    }
}

impl<'a, T> StridedIterator for StridedMut<'a, T> where T: CommonBounds {
    type Item = &'a mut T;
    fn for_each<F>(mut self, func: F) where F: Fn(Self::Item) {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size() + 1;
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(self.inner_loop_next(idx));
            }
            self.next();
        }
    }

    fn for_each_init<F, INIT, I>(mut self, init: INIT, func: F)
        where F: Fn(&mut I, Self::Item), INIT: Fn() -> I
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size() + 1;
        let mut init = init();
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(&mut init, self.inner_loop_next(idx));
            }
            self.next();
        }
    }
}

impl<'a, T: 'a> IterGetSet for StridedMut<'a, T> where T: CommonBounds {
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
            self.base.ptr
                .get_ptr()
                .add(index * (self.base.last_stride as usize))
                .as_mut()
                .unwrap()
        }
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.base.set_prg(prg);
    }
}
