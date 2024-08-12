use std::sync::Arc;

use tensor_common::{ axis::Axis, shape::Shape, strides::Strides };

pub trait IterGetSet {
    type Item;
    fn set_end_index(&mut self, end_index: usize);
    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>);
    fn set_strides(&mut self, last_stride: Strides);
    fn set_shape(&mut self, shape: Shape);
    fn set_prg(&mut self, prg: Vec<i64>);
    fn intervals(&self) -> &Arc<Vec<(usize, usize)>>;
    fn strides(&self) -> &Strides;
    fn shape(&self) -> &Shape;
    fn broadcast_set_strides(&mut self, shape: &Shape);
    fn outer_loop_size(&self) -> usize;
    fn inner_loop_size(&self) -> usize;
    fn next(&mut self);
    fn inner_loop_next(&mut self, index: usize) -> Self::Item;
}

pub trait ShapeManipulator {
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self;
    fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self;
    fn expand<S: Into<Shape>>(self, shape: S) -> Self;
}

pub trait StridedIterator where Self: Sized {
    type Item;
    fn for_each<F>(self, func: F) where F: Fn(Self::Item);
    fn for_each_init<F, INIT, T>(self, init: INIT, func: F)
        where F: Fn(&mut T, Self::Item), INIT: Fn() -> T;
}
