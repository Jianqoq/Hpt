use std::sync::Arc;

use tensor_common::{ axis::Axis, shape::Shape, strides::Strides };

pub trait IterGetSet {
    type Item;
    fn set_end_index(&mut self, end_index: usize);
    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>);
    fn set_strides(&mut self, last_stride: Strides);
    fn set_shape(&mut self, shape: Shape);
    fn intervals(&self) -> &Arc<Vec<(usize, usize)>>;
    fn strides(&self) -> &Strides;
    fn shape(&self) -> &Shape;
    fn broadcast_set_strides(&mut self, shape: &Shape);
}

pub trait ShapeManipulator {
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self;
    fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self;
    fn expand<S: Into<Shape>>(self, shape: S) -> Self;
}
