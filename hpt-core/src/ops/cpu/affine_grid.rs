use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, Barrier},
};

use hpt_common::{shape::shape::Shape, shape::shape_utils::mt_intervals};
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};
use hpt_types::{dtype::FloatConst, into_scalar::Cast};

use crate::{tensor_base::_Tensor, Tensor, THREAD_POOL};

impl<T> _Tensor<T>
where
    T: CommonBounds
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Add<Output = T>
        + Neg<Output = T>
        + FloatConst,
    i64: Cast<T>,
{
    /// Generates an affine grid for the given shape.
    ///
    /// This method creates an affine grid, commonly used in applications like
    /// spatial transformations. The output is a grid of coordinates that can
    /// be used to apply affine transformations on input tensors. It supports
    /// both aligned corners and non-aligned corners, depending on the
    /// `align_corners` flag.
    ///
    /// # Arguments
    ///
    /// * `shape` - The target shape for the affine grid. It can be provided
    ///   as a type that implements the `Into<Shape>` trait, allowing flexibility
    ///   in specifying the shape.
    /// * `align_corners` - A boolean flag that, if true, aligns the grid corners
    ///   with the corners of the input tensor. If false, the grid will be
    ///   aligned in a different way, typically for downscaling operations.
    pub fn affine_grid<S: Into<Shape>>(
        &self,
        shape: S,
        align_corners: bool,
    ) -> anyhow::Result<_Tensor<T>> {
        let shape: Vec<i64> = shape.into().inner().clone();
        let theta_strides = self.strides();
        let theta_shape = self.shape();
        if shape.len() == 4 {
            if self.ndim() != 3
                || theta_shape[0] != shape[0]
                || theta_shape[1] != 2i64
                || theta_shape[2] != 3i64
            {
                anyhow::bail!("Theta shape must be [n, 2, 3]");
            }
            let n = shape[0];
            let h = shape[2];
            let w = shape[3];
            let grid = _Tensor::<T>::zeros(vec![n, h, w, 2])?;
            THREAD_POOL.with_borrow_mut(|x| {
                let outer_loop_size = grid.size() / 2;
                let num_threads;
                if outer_loop_size < x.max_count() {
                    num_threads = outer_loop_size;
                } else {
                    num_threads = x.max_count();
                }
                let mut intervals = mt_intervals(outer_loop_size, num_threads);
                let mut ptrs = Vec::new();
                let mut prgs = Vec::new();
                for i in 0..num_threads {
                    let start = intervals[i].0;
                    let mut ptr = grid.ptr();
                    let mut curent_shape_prg = [0i64; 4];
                    let mut amount = start * 2;
                    let mut index = 0;
                    for j in (0..grid.ndim()).rev() {
                        curent_shape_prg[j] = (amount as i64) % grid.shape()[j];
                        amount /= grid.shape()[j] as usize;
                        index += curent_shape_prg[j] * grid.strides()[j];
                    }
                    prgs.push(curent_shape_prg);
                    ptr.offset(index);
                    ptrs.push(ptr);
                }
                let w: T = w.cast();
                let h: T = h.cast();
                let barrier = Arc::new(Barrier::new(num_threads + 1));
                for _ in 0..num_threads {
                    let ts0 = theta_strides[0];
                    let ts1 = theta_strides[1];
                    let ts2 = theta_strides[2];
                    let theta_ptr = self.ptr();
                    let (start, end) = intervals.pop().unwrap();
                    let mut ptr = ptrs.pop().unwrap();
                    let mut prg = prgs.pop().unwrap();
                    let layout = grid.layout().clone();
                    let barrier_clone = Arc::clone(&barrier);
                    if align_corners {
                        x.execute(move || {
                            for _ in start..end {
                                let ni = prg[0];
                                let hi: T = prg[1].cast();
                                let wi: T = prg[2].cast();
                                let x = (T::TWO * wi) / (w - T::ONE) - T::ONE;
                                let y = (T::TWO * hi) / (h - T::ONE) - T::ONE;
                                let tx = theta_ptr[ni * ts0 + 0 * ts1 + 0 * ts2] * x
                                    + theta_ptr[ni * ts0 + 0 * ts1 + 1 * ts2] * y
                                    + theta_ptr[ni * ts0 + 0 * ts1 + 2 * ts2];
                                let ty = theta_ptr[ni * ts0 + 1 * ts1 + 0 * ts2] * x
                                    + theta_ptr[ni * ts0 + 1 * ts1 + 1 * ts2] * y
                                    + theta_ptr[ni * ts0 + 1 * ts1 + 2 * ts2];
                                ptr.modify(0, tx);
                                ptr.modify(1, ty);

                                for j in (0..(layout.ndim() as i64) - 1).rev() {
                                    let j = j as usize;
                                    if prg[j] < layout.shape()[j] - 1 {
                                        prg[j] += 1;
                                        ptr.offset(layout.strides()[j]);
                                        break;
                                    } else {
                                        prg[j] = 0;
                                        ptr.offset(-layout.strides()[j] * (layout.shape()[j] - 1));
                                    }
                                }
                            }
                            barrier_clone.wait();
                        });
                    } else {
                        x.execute(move || {
                            for _ in start..end {
                                let ni = prg[0];
                                let hi: T = prg[1].cast();
                                let wi: T = prg[2].cast();
                                let x = -T::ONE + (T::TWO * (wi + T::HALF)) / w;
                                let y = -T::ONE + (T::TWO * (hi + T::HALF)) / h;
                                let tx = theta_ptr[ni * ts0] * x
                                    + theta_ptr[ni * ts0 + ts2] * y
                                    + theta_ptr[ni * ts0 + ts2 * 2];
                                let ty = theta_ptr[ni * ts0 + ts1] * x
                                    + theta_ptr[ni * ts0 + ts1 + ts2] * y
                                    + theta_ptr[ni * ts0 + ts1 + ts2 * 2];
                                ptr.modify(0, tx);
                                ptr.modify(1, ty);
                                for j in (0..(layout.ndim() as i64) - 1).rev() {
                                    let j = j as usize;
                                    if prg[j] < layout.shape()[j] - 1 {
                                        prg[j] += 1;
                                        ptr.offset(layout.strides()[j]);
                                        break;
                                    } else {
                                        prg[j] = 0;
                                        ptr.offset(-layout.strides()[j] * (layout.shape()[j] - 1));
                                    }
                                }
                            }
                            barrier_clone.wait();
                        });
                    }
                }
                barrier.wait();
            });
            Ok(grid)
        } else {
            anyhow::bail!("Shape must be 4D")
        }
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Add<Output = T>
        + Neg<Output = T>
        + FloatConst,
    i64: Cast<T>,
{
    /// Generates an affine grid for the given shape.
    ///
    /// This method creates an affine grid, commonly used in applications like
    /// spatial transformations. The output is a grid of coordinates that can
    /// be used to apply affine transformations on input tensors. It supports
    /// both aligned corners and non-aligned corners, depending on the
    /// `align_corners` flag.
    ///
    /// # Arguments
    ///
    /// * `shape` - The target shape for the affine grid. It can be provided
    ///   as a type that implements the `Into<Shape>` trait, allowing flexibility
    ///   in specifying the shape.
    /// * `align_corners` - A boolean flag that, if true, aligns the grid corners
    ///   with the corners of the input tensor. If false, the grid will be
    ///   aligned in a different way, typically for downscaling operations.
    pub fn affine_grid<S: Into<Shape>>(
        &self,
        shape: S,
        align_corners: bool,
    ) -> anyhow::Result<Tensor<T>> {
        Ok(self.inner.affine_grid(shape, align_corners)?.into())
    }
}
