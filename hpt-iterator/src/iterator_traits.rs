use std::sync::Arc;

use hpt_common::{
    axis::axis::Axis,
    layout::layout::Layout,
    shape::shape::Shape,
    shape::shape_utils::{mt_intervals, predict_broadcast_shape},
    strides::strides::Strides,
};
use hpt_traits::CommonBounds;
use rayon::iter::{plumbing::UnindexedProducer, ParallelIterator};

use crate::{
    par_strided_zip::{par_strided_zip_simd::ParStridedZipSimd, ParStridedZip},
    strided_map::StridedMap,
    strided_zip::{strided_zip_simd::StridedZipSimd, StridedZip},
    with_simd::WithSimd,
};

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
    /// get the layout of the iterator
    fn layout(&self) -> &Layout;
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
    /// get the layout of the iterator
    fn layout(&self) -> &Layout;
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
    fn inner_loop_next_simd(&mut self, index: usize) -> Self::SimdItem;
    /// check if all iterators' last stride is one, only when all iterators' last stride is one, we can do simd iteration
    fn all_last_stride_one(&self) -> bool;
    /// get the simd vector size, if any of the iterator returned different vector size, it will return None
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
pub trait StridedIterator: IterGetSet
where
    Self: Sized,
{
    /// perform scalar iteration, this method is for single thread iterator
    fn for_each<F>(mut self, func: F)
    where
        F: Fn(Self::Item),
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size(); // we don't need to add 1 as we didn't subtract shape by 1
        self.set_prg(vec![0; self.shape().len()]);
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(self.inner_loop_next(idx));
            }
            self.next();
        }
    }
    /// perform scalar iteration with init, this method is for single thread iterator
    fn for_each_init<F, INIT, T>(mut self, init: INIT, func: F)
    where
        F: Fn(&mut T, Self::Item),
        INIT: Fn() -> T,
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size();
        self.set_prg(vec![0; self.shape().len()]);
        let mut init = init();
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(&mut init, self.inner_loop_next(idx));
            }
            self.next();
        }
    }
}

/// A trait to zip two iterators together.
pub trait StridedIteratorZip: Sized {
    /// Combines this iterator with another iterator, enabling simultaneous iteration.
    ///
    /// This method zips together `self` and `other` into a `StridedZip` iterator, allowing for synchronized
    ///
    /// iteration over both iterators. This is particularly useful for operations that require processing
    ///
    /// elements from two tensors in parallel, such as element-wise arithmetic operations.
    ///
    /// # Arguments
    ///
    /// * `other` - The other iterator to zip with. It must implement the `IterGetSet` trait, and
    ///             its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A `StridedZip` instance that encapsulates both `self` and `other`, allowing for synchronized
    ///
    /// iteration over their elements.
    ///
    /// # Panics
    ///
    /// This method will panic if the shapes of `self` and `other` cannot be broadcasted together.
    #[track_caller]
    fn zip<'a, C>(self, other: C) -> StridedZip<'a, Self, C>
    where
        C: IterGetSet + ShapeManipulator,
        Self: IterGetSet + ShapeManipulator,
        <C as IterGetSet>::Item: Send,
        <Self as IterGetSet>::Item: Send,
    {
        let new_shape = predict_broadcast_shape(&self.shape(), &other.shape())
            .expect("Cannot broadcast shapes");

        let mut a = self.reshape(new_shape.clone());
        let mut b = other.reshape(new_shape.clone());

        a.set_shape(new_shape.clone());
        b.set_shape(new_shape.clone());
        StridedZip::new(a, b)
    }
}

/// A trait to zip two parallel iterators together.
pub trait ParStridedIteratorZip: Sized + IterGetSet {
    /// Combines this iterator with another iterator, enabling simultaneous parallel iteration.
    ///
    /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
    /// iterate over tensors with compatible shapes. It adjusts the strides and shapes of both iterators
    /// to match the broadcasted shape and then returns a `ParStridedZip` that allows for synchronized
    /// parallel iteration over both iterators.
    ///
    /// # Arguments
    ///
    /// * `other` - The other iterator to zip with. It must implement the `IterGetSet`, `UnindexedProducer`,
    ///             and `ParallelIterator` traits, and its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A `ParStridedZip` instance that zips together `self` and `other`, enabling synchronized
    /// parallel iteration over their elements.
    ///
    /// # Panics
    ///
    /// This method will panic if the shapes of `self` and `other` cannot be broadcasted together.
    /// Ensure that the shapes are compatible before calling this method.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zip<'a, C>(mut self, mut other: C) -> ParStridedZip<'a, Self, C>
    where
        C: UnindexedProducer + 'a + IterGetSet + ParallelIterator + ShapeManipulator,
        <C as IterGetSet>::Item: Send,
        Self: UnindexedProducer + ParallelIterator + ShapeManipulator,
        <Self as IterGetSet>::Item: Send,
    {
        let new_shape = predict_broadcast_shape(&self.shape(), &other.shape())
            .expect("Cannot broadcast shapes");

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
}

/// A trait to zip two parallel iterators together.
pub trait ParStridedIteratorSimdZip: Sized + IterGetSetSimd {
    /// Combines this `ParStridedZipSimd` iterator with another SIMD-optimized iterator, enabling simultaneous parallel iteration.
    ///
    /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
    /// iterate over tensors with compatible shapes. It calculates the appropriate iteration intervals based
    /// on the new broadcasted shape and configures both iterators accordingly. Finally, it returns a new
    /// `ParStridedZipSimd` instance that allows for synchronized parallel iteration over the combined iterators.
    ///
    /// # Arguments
    ///
    /// * `other` - The third iterator to zip with. It must implement the `IterGetSetSimd`, `UnindexedProducer`,
    ///             `ShapeManipulator`, and `ParallelIterator` traits,
    ///             and its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A new `ParStridedZipSimd` instance that combines `self` and `other` for synchronized parallel iteration over all three iterators.
    ///
    /// # Panics
    ///
    /// This method will panic if the shapes of `self` and `other` cannot be broadcasted together.
    /// Ensure that the shapes are compatible before calling this method.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zip<'a, C>(mut self, mut other: C) -> ParStridedZipSimd<'a, Self, C>
    where
        C: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator + ShapeManipulator,
        <C as IterGetSetSimd>::Item: Send,
        Self: UnindexedProducer + ParallelIterator + ShapeManipulator,
        <Self as IterGetSetSimd>::Item: Send,
    {
        let new_shape = predict_broadcast_shape(&self.shape(), &other.shape())
            .expect("Cannot broadcast shapes");

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

        ParStridedZipSimd::new(a, b)
    }
}

/// A trait to zip two simd iterators together.
pub trait StridedSimdIteratorZip: Sized {
    /// Combines this iterator with another SIMD-optimized iterator, enabling simultaneous iteration.
    ///
    /// This method performs shape broadcasting between `self` and `other` to ensure that both iterators
    /// iterate over tensors with compatible shapes. It adjusts the strides and shapes of both iterators
    /// to match the broadcasted shape and then returns a `StridedZipSimd` that allows for synchronized
    /// iteration over both iterators.
    ///
    /// # Arguments
    ///
    /// * `other` - The other iterator to zip with. It must implement the `IterGetSetSimd`, `UnindexedProducer`,
    ///             and `ParallelIterator` traits, and its associated `Item` type must be `Send`.
    ///
    /// # Returns
    ///
    /// A `StridedZipSimd` instance that zips together `self` and `other`, enabling synchronized
    /// iteration over their elements.
    #[track_caller]
    fn zip<'a, C>(mut self, mut other: C) -> StridedZipSimd<'a, Self, C>
    where
        C: 'a + IterGetSetSimd,
        <C as IterGetSetSimd>::Item: Send,
        Self: IterGetSetSimd,
        <Self as IterGetSetSimd>::Item: Send,
    {
        let new_shape =
            predict_broadcast_shape(self.shape(), other.shape()).expect("Cannot broadcast shapes");

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        StridedZipSimd::new(self, other)
    }
}

/// A trait for performing single thread simd iteration over an iterator.
pub trait StridedIteratorSimd
where
    Self: Sized + IterGetSetSimd,
{
    /// perform simd iteration, this method is for single thread simd iterator
    fn for_each<F, F2>(mut self, op: F, vec_op: F2)
    where
        F: Fn(Self::Item),
        F2: Fn(Self::SimdItem),
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size(); // we don't need to add 1 as we didn't subtract shape by 1
        self.set_prg(vec![0; self.shape().len()]);
        match (self.all_last_stride_one(), self.lanes()) {
            (true, Some(vec_size)) => {
                let remain = inner_loop_size % vec_size;
                let inner = inner_loop_size - remain;
                let n = inner / vec_size;
                let unroll = n % 4;
                if remain > 0 {
                    if unroll == 0 {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n / 4 {
                                vec_op(self.inner_loop_next_simd(idx * 4));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 1));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 2));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 3));
                            }
                            for idx in inner..inner_loop_size {
                                op(self.inner_loop_next(idx));
                            }
                            self.next();
                        }
                    } else {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n {
                                vec_op(self.inner_loop_next_simd(idx));
                            }
                            for idx in inner..inner_loop_size {
                                op(self.inner_loop_next(idx));
                            }
                            self.next();
                        }
                    }
                } else {
                    if unroll == 0 {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n / 4 {
                                vec_op(self.inner_loop_next_simd(idx * 4));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 1));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 2));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 3));
                            }
                            self.next();
                        }
                    } else {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n {
                                vec_op(self.inner_loop_next_simd(idx));
                            }
                            self.next();
                        }
                    }
                }
            }
            _ => {
                for _ in 0..outer_loop_size {
                    for idx in 0..inner_loop_size {
                        op(self.inner_loop_next(idx));
                    }
                    self.next();
                }
            }
        }
    }
    /// perform simd iteration with init, this method is for single thread simd iterator
    fn for_each_init<F, INIT, T>(mut self, init: INIT, func: F)
    where
        F: Fn(&mut T, Self::Item),
        INIT: Fn() -> T,
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size();
        let mut init = init();
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(&mut init, self.inner_loop_next(idx));
            }
            self.next();
        }
    }
}

/// A trait for performing single thread simd iteration over an iterator.
pub trait ParStridedIteratorSimd
where
    Self: Sized + UnindexedProducer + IterGetSetSimd + ParallelIterator,
{
    /// perform simd iteration, this method is for single thread simd iterator
    fn for_each<F, F2>(self, op: F, vec_op: F2)
    where
        F: Fn(<Self as IterGetSetSimd>::Item) + Sync,
        F2: Fn(<Self as IterGetSetSimd>::SimdItem) + Sync + Send + Copy,
        <Self as IterGetSetSimd>::SimdItem: Send,
        <Self as IterGetSetSimd>::Item: Send,
    {
        let with_simd = WithSimd { base: self, vec_op };
        with_simd.for_each(|x| {
            op(x);
        });
    }
}

/// A trait to map a function on the elements of an iterator.
pub trait StridedIteratorMap: Sized {
    /// Transforms the strided iterators by applying a provided function to their items.
    ///
    /// This method allows for element-wise operations on the zipped iterators by applying `func` to each item.
    ///
    /// # Type Parameters
    ///
    /// * `'a` - The lifetime associated with the iterators.
    /// * `F` - The function to apply to each item.
    /// * `U` - The output type after applying the function.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an item from the zipped iterator and returns a transformed value.
    ///
    /// # Returns
    ///
    /// A `StridedMap` instance that applies the provided function during iteration.
    fn map<'a, T, F, U>(self, f: F) -> StridedMap<'a, Self, T, F>
    where
        F: Fn(T) -> U + Sync + Send + 'a,
        U: CommonBounds,
        Self: IterGetSet<Item = T>,
    {
        StridedMap {
            iter: self,
            f,
            phantom: std::marker::PhantomData,
        }
    }
}

pub(crate) trait StridedHelper {
    fn _set_last_strides(&mut self, stride: i64);
    fn _set_strides(&mut self, strides: Strides);
    fn _set_shape(&mut self, shape: Shape);
    fn _layout(&self) -> &Layout;
}

pub(crate) trait ParStridedHelper {
    fn _set_last_strides(&mut self, stride: i64);
    fn _set_strides(&mut self, strides: Strides);
    fn _set_shape(&mut self, shape: Shape);
    fn _layout(&self) -> &Layout;
    fn _set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>);
    fn _set_end_index(&mut self, end_index: usize);
}
