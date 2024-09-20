use std::sync::Arc;

use tensor_common::{ axis::Axis, shape::Shape, strides::Strides };

/// A trait for getting and setting values from an iterator.
pub trait IterGetSet {
    /// The type of the iterator's elements.
    type Item;
    /// set the end index of the iterator, this is used when rayon perform data splitting
    fn set_end_index(&mut self, end_index: usize);
    /// set the chunk intervals of the iterator, we chunk the outer loop
    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>);
    /// set the strides for the iterator, we call this method normally when we do broadcasting
    fn set_strides(&mut self, strides: Strides);
    /// set the shape for the iterator, we call this method normally when we do broadcasting
    fn set_shape(&mut self, shape: Shape);
    /// set the loop progress for the iterator
    fn set_prg(&mut self, prg: Vec<i64>);
    /// get the intervals of the iterator
    fn intervals(&self) -> &Arc<Vec<(usize, usize)>>;
    /// get the strides of the iterator
    fn strides(&self) -> &Strides;
    /// get the shape of the iterator
    fn shape(&self) -> &Shape;
    /// set the strides for all the iterators
    fn broadcast_set_strides(&mut self, shape: &Shape);
    /// get the outer loop size
    fn outer_loop_size(&self) -> usize;
    /// get the inner loop size
    fn inner_loop_size(&self) -> usize;
    /// update the loop progress
    fn next(&mut self);
    /// get the next element of the inner loop
    fn inner_loop_next(&mut self, index: usize) -> Self::Item;
}

/// A trait for getting and setting values from an simd iterator
pub trait IterGetSetSimd {
    /// The type of the iterator's elements.
    type Item;
    /// The type of the iterator's simd elements.
    type SimdItem;
    /// set the end index of the iterator, this is used when rayon perform data splitting
    fn set_end_index(&mut self, end_index: usize);
    /// set the chunk intervals of the iterator, we chunk the outer loop
    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>);
    /// set the strides for the iterator, we call this method normally when we do broadcasting
    fn set_strides(&mut self, last_stride: Strides);
    /// set the shape for the iterator, we call this method normally when we do broadcasting
    fn set_shape(&mut self, shape: Shape);
    /// set the loop progress for the iterator
    fn set_prg(&mut self, prg: Vec<i64>);
    /// get the intervals of the iterator
    fn intervals(&self) -> &Arc<Vec<(usize, usize)>>;
    /// get the strides of the iterator
    fn strides(&self) -> &Strides;
    /// get the shape of the iterator
    fn shape(&self) -> &Shape;
    /// set the strides for all the iterators
    fn broadcast_set_strides(&mut self, shape: &Shape);
    /// get the outer loop size
    fn outer_loop_size(&self) -> usize;
    /// get the inner loop size
    fn inner_loop_size(&self) -> usize;
    /// update the loop progress, this is called when we don't do simd iteration
    fn next(&mut self);
    /// update the loop progress, this is called when we do simd iteration
    fn next_simd(&mut self);
    /// get the next element of the inner loop
    fn inner_loop_next(&mut self, index: usize) -> Self::Item;
    /// get the next vector of the inner loop, this is called when we do simd iteration
    fn inner_loop_next_simd(&self, index: usize) -> Self::SimdItem;
    /// check if all iterators' last stride is one, only when all iterators' last stride is one, we can do simd iteration
    fn all_last_stride_one(&self) -> bool;
    /// get the simd vector size
    fn lanes(&self) -> Option<usize>;
}

/// A trait for performing shape manipulation on an iterator.
pub trait ShapeManipulator {
    /// reshape the iterator, we can change the iteration behavior by changing the shape
    fn reshape<S: Into<Shape>>(self, shape: S) -> Self;
    /// transpose the iterator, we can change the iteration behavior by changing the axes
    fn transpose<AXIS: Into<Axis>>(self, axes: AXIS) -> Self;
    /// expand the iterator, we can change the iteration behavior by changing the shape
    fn expand<S: Into<Shape>>(self, shape: S) -> Self;
}

/// A trait for performing single thread iteration over an iterator.
pub trait StridedIterator where Self: Sized {
    /// The type of the iterator's elements.
    type Item;
    /// perform scalar iteration, this method is for single thread iterator
    fn for_each<F>(self, func: F) where F: Fn(Self::Item);
    /// perform scalar iteration with init, this method is for single thread iterator
    fn for_each_init<F, INIT, T>(self, init: INIT, func: F)
        where F: Fn(&mut T, Self::Item), INIT: Fn() -> T;
}

/// A trait for performing single thread simd iteration over an iterator.
pub trait StridedIteratorSimd where Self: Sized {
    /// The type of the iterator's elements.
    type Item;
    /// The type of the iterator's simd elements.
    type SimdItem;
    /// perform simd iteration, this method is for single thread simd iterator
    fn for_each<F, F2>(self, func: F, func2: F2) where F: Fn(Self::Item), F2: Fn(Self::SimdItem);
    /// perform simd iteration with init, this method is for single thread simd iterator
    fn for_each_init<F, INIT, T>(self, init: INIT, func: F)
        where F: Fn(&mut T, Self::Item), INIT: Fn() -> T;
}
