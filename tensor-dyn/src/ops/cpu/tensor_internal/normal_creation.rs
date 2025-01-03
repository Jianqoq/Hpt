use std::{panic::Location, sync::Arc};

use crate::{
    backend::{Backend, Cpu},
    ops::common::creation::geomspace_preprocess_start_step,
    tensor_base::_Tensor,
    BoolVector, ALIGN,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tensor_allocator::{traits::Allocator, CACHE};
use tensor_common::error::memory::MemoryError;
use tensor_common::{
    error::{base::TensorError, shape::ShapeError},
    layout::Layout,
    pointer::Pointer,
    shape::Shape,
};
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo, TensorLike};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    into_scalar::IntoScalar,
    type_promote::NormalOut,
};

impl<T: CommonBounds, const DEVICE: usize> TensorCreator<T> for _Tensor<T, Cpu, DEVICE> {
    type Output = _Tensor<T, Cpu, DEVICE>;
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
        let ptr = CACHE
            .lock()
            .expect("CACHE is poisoned")
            .allocate(layout, DEVICE)?;
        Ok(_Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, size as i64),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::from(res_shape.clone()),
            mem_layout: Arc::new(layout),
            _backend: Backend::<Cpu>::new(ptr as u64),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        Self::full(T::ZERO, shape)
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self, TensorError>
    where
        u8: IntoScalar<T>,
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
        u8: IntoScalar<T>,
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
        _Tensor::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> Result<Self, TensorError>
    where
        T: Convertor + FromScalar<U>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        let size = end.to_i64() - start.to_i64();
        let start = start.into_scalar();
        if size <= 0 {
            return _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(vec![0]));
        }
        let mut data: _Tensor<T, Cpu, DEVICE> =
            _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(vec![size]))?;

        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.into_scalar());
            });
        Ok(data)
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self, TensorError>
    where
        T: Convertor + FromScalar<usize>,
    {
        let step_float = step.to_f64();
        let end_usize = end.to_i64();
        let start_usize = start.to_i64();
        let size = ((end_usize - start_usize) as usize) / (step_float.abs() as usize);
        let mut data = _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(vec![size as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(T::_from(i)._mul(step));
            });
        Ok(data)
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self, TensorError>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(shape))?;
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

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> Result<Self, TensorError>
    where
        T: Convertor,
        U: Convertor + IntoScalar<T> + Copy,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        let _start: f64 = start.to_f64();
        let _end: f64 = end.to_f64();
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.into_scalar();
        let start_t: T = start.into_scalar();
        let end_t: T = end.into_scalar();
        let mut data = _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                if include_end && i == num - 1 {
                    *x = end_t;
                } else {
                    *x = start_t._add(i.into_scalar()._mul(step_t));
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
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64>,
    {
        let _start = start.to_f64();
        let _end = end.to_f64();
        let n = num as f64;
        let step = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t = T::_from(step);
        let mut data = _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = base._pow(start._add(T::_from(i)._mul(step_t)));
            });
        Ok(data)
    }

    fn geomspace(
        start: T,
        end: T,
        n: usize,
        include_end: bool,
    ) -> Result<Self, TensorError>
    where
        f64: IntoScalar<T>,
        usize: IntoScalar<T>,
    {
        let start_f64 = start.to_f64();
        let end_f64 = end.to_f64();
        let both_negative = start_f64 < 0.0 && end_f64 < 0.0;
        let (new_start, step) =
            geomspace_preprocess_start_step(start_f64, end_f64, n, include_end)?;
        let start_t: T = new_start.into_scalar();
        let step_t: T = step.into_scalar();
        let mut data = _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(vec![n as i64]))?;
        if both_negative {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let i: T = i.into_scalar();
                    let val: T = T::TEN._pow(start_t._add(i._mul(step_t)));
                    *x = val._neg();
                });
        } else {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let i: T = i.into_scalar();
                    let val: T = T::TEN._pow(start_t._add(i._mul(step_t)));
                    *x = val;
                });
        }
        Ok(data)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self, TensorError>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = _Tensor::<T, Cpu, DEVICE>::empty(Arc::new(shape))?;
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
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        ShapeError::check_ndim_enough(2, self.shape().len())?;
        let mask: _Tensor<bool, Cpu, DEVICE> = _Tensor::<bool, Cpu, DEVICE>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            true,
        )?;
        let res: _Tensor<T, Cpu, DEVICE> = self.clone() * mask;
        Ok(res)
    }

    fn triu(&self, k: i64) -> Result<Self, TensorError>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        ShapeError::check_ndim_enough(2, self.shape().len())?;
        let mask: _Tensor<bool, Cpu, DEVICE> = _Tensor::<bool, Cpu, DEVICE>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            false,
        )?;
        let res: _Tensor<T, Cpu, DEVICE> = self.clone() * mask;
        Ok(res)
    }

    fn identity(n: usize) -> Result<Self, TensorError>
    where
        u8: IntoScalar<T>,
    {
        _Tensor::<T, Cpu, DEVICE>::eye(n, n, 0)
    }
}
