use std::{fmt::Display, sync::Arc};

use rayon::iter::{
    plumbing::{ bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer },
    ParallelIterator,
};
use tensor_common::{ pointer::Pointer, shape::Shape, shape_utils::mt_intervals, strides::Strides };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };

use crate::{ iterator_traits::IterGetSet, strided_zip::StridedZip };

#[derive(Debug)]
pub struct StridedMapMut<'a, T> where T: Copy {
    pub(crate) ptr: Pointer<T>,
    pub(crate) intervals: Arc<Vec<(usize, usize)>>,
    pub(crate) shape: Shape,
    pub(crate) start_index: usize,
    pub(crate) end_index: usize,
    pub(crate) last_stride: i64,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> StridedMapMut<'a, T> where T: CommonBounds {
    pub fn new<U: TensorInfo<T>>(res_tensor: U) -> Self {
        let inner_loop_size = res_tensor.shape()[res_tensor.shape().len() - 1] as usize;
        let outer_loop_size = res_tensor.size() / (inner_loop_size);
        let mut num_threads = rayon::current_num_threads();
        if outer_loop_size < num_threads {
            num_threads = outer_loop_size;
        }
        let intervals: Vec<(usize, usize)> = mt_intervals(outer_loop_size, num_threads);
        let len = intervals.len();
        StridedMapMut {
            ptr: res_tensor.ptr(),
            shape: res_tensor.shape().clone(),
            intervals: Arc::new(intervals),
            start_index: 0,
            end_index: len,
            last_stride: res_tensor.strides()[res_tensor.strides().len() - 1],
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zip<C>(self, other: C) -> StridedZip<'a, Self, C>
        where
            C: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
            <C as IterGetSet>::Item: Send
    {
        StridedZip::new(self, other)
    }
}

impl<'a, T> ParallelIterator for StridedMapMut<'a, T> where T: Clone + Sync + Send + 'a + Copy {
    type Item = &'a mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where C: UnindexedConsumer<Self::Item> {
        bridge_unindexed(self, consumer)
    }
}

impl<'a, T> UnindexedProducer for StridedMapMut<'a, T> where T: Clone + Sync + Send + 'a + Copy {
    type Item = &'a mut T;

    fn split(mut self) -> (Self, Option<Self>) {
        if self.end_index - self.start_index <= 1 {
            let index = self.intervals[self.start_index].0 * (*self.shape.last().unwrap() as usize);
            self.ptr.add(index);
            return (self, None);
        }
        let _left_interval = &self.intervals[self.start_index..self.end_index];
        let left = _left_interval.len() / 2;
        let right = _left_interval.len() / 2 + (_left_interval.len() % 2);
        (
            StridedMapMut {
                ptr: self.ptr,
                shape: self.shape.clone(),
                intervals: self.intervals.clone(),
                start_index: self.start_index,
                end_index: self.start_index + left,
                last_stride: self.last_stride,
                phantom: std::marker::PhantomData,
            },
            Some(StridedMapMut {
                ptr: self.ptr,
                shape: self.shape.clone(),
                intervals: self.intervals.clone(),
                start_index: self.start_index + left,
                end_index: self.start_index + left + right,
                last_stride: self.last_stride,
                phantom: std::marker::PhantomData,
            }),
        )
    }

    fn fold_with<F>(self, folder: F) -> F where F: Folder<Self::Item> {
        folder
    }
}

impl<'a, T: 'a + Display + Copy> IterGetSet for StridedMapMut<'a, T> {
    type Item = &'a mut T;

    fn set_end_index(&mut self, end_index: usize) {
        self.end_index = end_index;
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.intervals = intervals;
    }

    fn set_strides(&mut self, _: Strides) {}

    fn set_shape(&mut self, shape: Shape) {
        self.shape = shape;
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        &self.intervals
    }

    fn strides(&self) -> &Strides {
        unreachable!()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn broadcast_set_strides(&mut self, _: &Shape) {}

    fn outer_loop_size(&self) -> usize {
        self.intervals[self.start_index].1 - self.intervals[self.start_index].0
    }

    fn inner_loop_size(&self) -> usize {
        self.shape[self.shape.len() - 1] as usize
    }

    fn next(&mut self) {
        self.ptr.offset(self.inner_loop_size() as i64);
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        unsafe { self.ptr.get_ptr().add(index).as_mut().unwrap() }
    }
}
