use std::{
    ops::{Div, Sub},
    panic::Location,
    sync::Arc,
};

use crate::{
    backend::{Backend, Cpu},
    tensor_base::_Tensor,
    BoolVector, ALIGN,
};
use anyhow::Result;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tensor_allocator::CACHE;
use tensor_common::{err_handler::ErrHandler, layout::Layout, pointer::Pointer, shape::Shape};
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo, TensorLike};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::Dtype,
    into_scalar::IntoScalar,
    type_promote::{FloatOutUnary, NormalOut},
};

impl<T: CommonBounds> TensorCreator<T> for _Tensor<T> {
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let mut size = 1;
        let mut strides = vec![0; res_shape.len()];
        for i in (0..res_shape.len()).rev() {
            let tmp = res_shape[i];
            strides[i] = size as i64;
            size *= tmp as usize;
        }
        let layout = std::alloc::Layout::from_size_align(size * size_of::<T>(), ALIGN)?;
        let ptr = CACHE.allocate(layout);
        let ly = Layout::new(res_shape.clone(), strides.clone());
        Ok(_Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, size as i64),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: ly,
            mem_layout: Arc::new(layout),
            _backend: Backend::new(ptr as u64),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Self::full(T::ZERO, shape)
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Self::full(T::ONE, shape)
    }

    fn empty_like(&self) -> Result<Self> {
        Self::empty(self.shape())
    }

    fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape())
    }

    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Self::ones(self.shape())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
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

    fn full_like(&self, val: T) -> Result<Self> {
        _Tensor::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> Result<Self>
    where
        T: Convertor + FromScalar<U>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        let size = end.to_i64() - start.to_i64();
        let start = start.into_scalar();
        if size <= 0 {
            return _Tensor::<T, Cpu>::empty(Arc::new(vec![0]));
        }
        let mut data: _Tensor<T> = _Tensor::<T, Cpu>::empty(Arc::new(vec![size]))?;

        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.into_scalar());
            });
        Ok(data)
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self>
    where
        T: Convertor + FromScalar<usize>,
    {
        let step_float = step.to_f64();
        let end_usize = end.to_i64();
        let start_usize = start.to_i64();
        let size = ((end_usize - start_usize) as usize) / (step_float.abs() as usize);
        let mut data = _Tensor::<T, Cpu>::empty(Arc::new(vec![size as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(T::__from(i)._mul(step));
            });
        Ok(data)
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = _Tensor::<T, Cpu>::empty(Arc::new(shape))?;
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

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float,
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
        let mut data = _Tensor::<T, Cpu>::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.into_scalar()._mul(step_t));
            });
        Ok(data)
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
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
        let step_t = T::__from(step);
        let mut data = _Tensor::<T, Cpu>::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = base._pow(start._add(T::__from(i)._mul(step_t)));
            });
        Ok(data)
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        T: PartialOrd + FromScalar<<T as FloatOutUnary>::Output> + std::ops::Neg<Output = T>,
        <T as FloatOutUnary>::Output: Sub<Output = <T as FloatOutUnary>::Output>
            + FromScalar<usize>
            + FromScalar<f64>
            + Div<Output = <T as FloatOutUnary>::Output>
            + CommonBounds,
    {
        let both_negative = start < T::ZERO && end < T::ZERO;
        let float_n = <T as FloatOutUnary>::Output::__from(n);
        let step = if include_end {
            if start > T::ZERO && end > T::ZERO {
                (end._log10() - start._log10())
                    / (float_n - <T as FloatOutUnary>::Output::__from(1f64))
            } else if start < T::ZERO && end < T::ZERO {
                (end._abs()._log10() - start._abs()._log10())
                    / (float_n - <T as FloatOutUnary>::Output::__from(1.0))
            } else {
                return Err(anyhow::Error::msg("start and end must have the same sign"));
            }
        } else if start > T::ZERO && end > T::ZERO {
            (end._log10() - start._log10()) / <T as FloatOutUnary>::Output::__from(n)
        } else if start < T::ZERO && end < T::ZERO {
            (end._abs()._log10() - start._abs()._log10()) / float_n
        } else {
            return Err(anyhow::Error::msg("start and end must have the same sign"));
        };
        let mut data = _Tensor::<T>::empty(Arc::new(vec![n as i64]))?;
        let ten: <T as FloatOutUnary>::Output = <T as FloatOutUnary>::Output::__from(10.0);
        let start = if start > T::ZERO {
            start._log10()
        } else {
            start._abs()._log10()
        };
        if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
            if both_negative {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten
                            ._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                        *x = -T::__from(val);
                    });
            } else {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten
                            ._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                        *x = T::__from(val);
                    });
            }
            return Ok(data);
        } else if both_negative {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let val =
                        ten._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                    *x = -T::__from(val);
                });
        } else {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let val =
                        ten._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                    *x = T::__from(val);
                });
        }
        Ok(data)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = _Tensor::<T, Cpu>::empty(Arc::new(shape))?;
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

    fn tril(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        if self.shape().len() < 2 {
            return Err(
                ErrHandler::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        let mask: _Tensor<bool> = _Tensor::<bool>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            true,
        )?;
        let res: _Tensor<T> = self.clone() * mask;
        Ok(res)
    }

    fn triu(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        if self.shape().len() < 2 {
            return Err(
                ErrHandler::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        let mask: _Tensor<bool> = _Tensor::<bool>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            false,
        )?;
        let res = self.clone() * mask;
        Ok(res)
    }

    fn identity(n: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        _Tensor::eye(n, n, 0)
    }
}
