use std::marker::PhantomData;
use std::{panic::Location, sync::Arc};

use crate::backend::Cpu;
use crate::ops::TensorCreator;
use crate::{
    backends::common::creation::geomspace_preprocess_start_step, tensor_base::_Tensor, BoolVector,
    ALIGN,
};
use hpt_allocator::traits::Allocator;
use hpt_allocator::traits::AllocatorOutputRetrive;
use hpt_allocator::Backend;
use hpt_common::error::memory::MemoryError;
use hpt_common::{
    error::{base::TensorError, shape::ShapeError},
    layout::layout::Layout,
    shape::shape::Shape,
    utils::pointer::Pointer,
};
use hpt_traits::tensor::{CommonBounds, TensorInfo, TensorLike};
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

impl<T, const DEVICE: usize, A> TensorCreator for _Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
    T: CommonBounds,
{
    type Output = _Tensor<T, Cpu, DEVICE, A>;
    type Meta = T;
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let size = res_shape
            .iter()
            .try_fold(1i64, |acc, &num| acc.checked_mul(num).or(Some(i64::MAX)))
            .unwrap_or(i64::MAX) as usize;
        let layout = std::alloc::Layout::from_size_align(
            size.checked_mul(size_of::<T>())
                .unwrap_or((isize::MAX as usize) - (ALIGN - 1)), // when overflow happened, we use max memory `from_size_align` accept
            ALIGN,
        )
        .map_err(|e| {
            TensorError::Memory(MemoryError::AllocationFailed {
                device: "cpu".to_string(),
                id: DEVICE,
                size,
                source: Some(Box::new(e)),
                location: Location::caller(),
            })
        })?;
        let mut allocator = A::new();
        let allocate_res = allocator.allocate(layout, DEVICE)?;
        let ptr = allocate_res.get_ptr();
        Ok(_Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, size as i64),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::from(res_shape.clone()),
            mem_layout: Arc::new(layout),
            _backend: Backend::<Cpu>::new(ptr as u64, DEVICE),
            phantom: PhantomData,
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let size = res_shape
            .iter()
            .try_fold(1i64, |acc, &num| acc.checked_mul(num).or(Some(i64::MAX)))
            .unwrap_or(i64::MAX) as usize;
        let layout = std::alloc::Layout::from_size_align(
            size.checked_mul(size_of::<T>())
                .unwrap_or((isize::MAX as usize) - (ALIGN - 1)), // when overflow happened, we use max memory `from_size_align` accept
            ALIGN,
        )
        .map_err(|e| {
            TensorError::Memory(MemoryError::AllocationFailed {
                device: "cpu".to_string(),
                id: DEVICE,
                size,
                source: Some(Box::new(e)),
                location: Location::caller(),
            })
        })?;
        let mut allocator = A::new();
        let allocate_res = allocator.allocate_zeroed(layout, DEVICE)?;
        let ptr = allocate_res.get_ptr();
        Ok(_Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, size as i64),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::from(res_shape.clone()),
            mem_layout: Arc::new(layout),
            _backend: Backend::<Cpu>::new(ptr as u64, DEVICE),
            phantom: PhantomData,
        })
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        Self::full(T::ONE, shape)
    }

    fn empty_like(&self) -> Result<Self, TensorError> {
        Self::empty(self.shape())
    }

    fn zeros_like(&self) -> Result<Self, TensorError> {
        Self::zeros(self.shape())
    }

    fn ones_like(&self) -> Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        Self::ones(self.shape())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self, TensorError> {
        let empty = Self::empty(shape)?;
        let ptr = empty.ptr().ptr;
        let size = empty.size();
        let mem_size = empty.mem_layout.size() / size_of::<T>();
        assert_eq!(size, mem_size);
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, size) };
        slice.into_par_iter().for_each(|x| {
            *x = val;
        });
        Ok(empty)
    }

    fn full_like(&self, val: T) -> Result<Self, TensorError> {
        Self::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> Result<Self, TensorError>
    where
        usize: Cast<T>,
        U: Cast<i64> + Cast<T> + Copy,
    {
        let end_i64: i64 = end.cast();
        let start_i64: i64 = start.cast();
        let size: i64 = end_i64 - start_i64;
        let start: T = start.cast();
        if size <= 0 {
            return Self::empty(Arc::new(vec![0]));
        }
        let mut data: Self = Self::empty(Arc::new(vec![size]))?;

        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.cast());
            });
        Ok(data)
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self, TensorError>
    where
        T: Cast<f64> + Cast<f64>,
        f64: Cast<T>,
        usize: Cast<T>,
    {
        let step_float: f64 = step.cast();
        let end_float: f64 = end.cast();
        let start_float: f64 = start.cast();
        let size = if step_float > 0.0 {
            ((end_float - start_float) / step_float).floor() as i64 + 1
        } else {
            ((start_float - end_float) / (-step_float)).floor() as i64 + 1
        };
        let mut data = Self::empty(Arc::new(vec![size as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.cast()._mul(step));
            });
        Ok(data)
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self, TensorError> {
        let shape = vec![n as i64, m as i64];
        let mut res = Self::empty(Arc::new(shape))?;
        res.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                let row = i / m;
                let col = i % m;
                if col == row + k {
                    *x = T::ONE;
                } else {
                    *x = T::ZERO;
                }
            });
        Ok(res)
    }

    fn linspace<U>(start: U, end: U, num: usize, include_end: bool) -> Result<Self, TensorError>
    where
        U: Cast<f64> + Cast<T> + Copy,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let _start: f64 = start.cast();
        let _end: f64 = end.cast();
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.cast();
        let start_t: T = start.cast();
        let end_t: T = end.cast();
        let mut data = Self::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                if include_end && i == num - 1 {
                    *x = end_t;
                } else {
                    *x = start_t._add(i.cast()._mul(step_t));
                }
            });
        Ok(data)
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T,
    ) -> Result<Self, TensorError>
    where
        T: Cast<f64> + num::Float + FloatOutBinary<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let _start: f64 = start.cast();
        let _end: f64 = end.cast();
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.cast();
        let mut data = Self::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = base._pow(start._add(i.cast()._mul(step_t)));
            });
        Ok(data)
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64> + FloatOutBinary<T, Output = T>,
    {
        let start_f64: f64 = start.cast();
        let end_f64: f64 = end.cast();
        let both_negative = start_f64 < 0.0 && end_f64 < 0.0;
        let (new_start, step) =
            geomspace_preprocess_start_step(start_f64, end_f64, n, include_end)?;
        let start_t: T = new_start.cast();
        let step_t: T = step.cast();
        let mut data = Self::empty(Arc::new(vec![n as i64]))?;
        if both_negative {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let i: T = i.cast();
                    let val: T = T::TEN._pow(start_t._add(i._mul(step_t)));
                    *x = val._neg();
                });
        } else {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let i: T = i.cast();
                    let val: T = T::TEN._pow(start_t._add(i._mul(step_t)));
                    *x = val;
                });
        }
        Ok(data)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = Self::empty(Arc::new(shape))?;
        if low_triangle {
            res.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let row = i / m;
                    let col = i % m;
                    if (col as i64) <= (row as i64) + k {
                        *x = T::ONE;
                    } else {
                        *x = T::ZERO;
                    }
                });
        } else {
            let k = k - 1;
            res.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let row = i / m;
                    let col = i % m;
                    if (col as i64) <= (row as i64) + k {
                        *x = T::ZERO;
                    } else {
                        *x = T::ONE;
                    }
                });
        }
        Ok(res)
    }

    fn tril(&self, k: i64) -> Result<Self, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        ShapeError::check_ndim_enough(
            "Tril expected 2 dimensions.".to_string(),
            2,
            self.shape().len(),
        )?;
        let mask: _Tensor<bool, Cpu, DEVICE, A> = _Tensor::<bool, Cpu, DEVICE, A>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            true,
        )?;
        let res: _Tensor<T, Cpu, DEVICE, A> = self.clone() * mask;
        Ok(res)
    }

    fn triu(&self, k: i64) -> Result<Self, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        ShapeError::check_ndim_enough(
            "Triu expected 2 dimensions.".to_string(),
            2,
            self.shape().len(),
        )?;
        let mask: _Tensor<bool, Cpu, DEVICE, A> = _Tensor::<bool, Cpu, DEVICE, A>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            false,
        )?;
        let res: _Tensor<T, Cpu, DEVICE, A> = self.clone() * mask;
        Ok(res)
    }

    fn identity(n: usize) -> Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        Self::eye(n, n, 0)
    }
}
