use std::{cell::RefCell, rc::Rc};

use crate::{
    backend::Cpu,
    tensor::{DiffTensor, Tensor},
    tensor_base::_Tensor,
    BoolVector,
};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{CommonBounds, TensorCreator};
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::NormalOut};

impl<T: CommonBounds, const DEVICE: usize> TensorCreator<T> for Tensor<T, Cpu, DEVICE> {
    type Output = Tensor<T, Cpu, DEVICE>;

    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::ones(shape)?.into())
    }

    fn empty_like(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::empty_like(self.inner.as_ref())?.into())
    }

    fn zeros_like(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::zeros_like(self.inner.as_ref())?.into())
    }

    fn ones_like(&self) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::ones_like(self.inner.as_ref())?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::full_like(self.inner.as_ref(), val)?.into())
    }

    fn arange<U>(start: U, end: U) -> Result<Self::Output, TensorError>
    where
        usize: Cast<T>,
        U: Cast<i64> + Cast<T> + Copy,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + Cast<usize>,
        usize: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::eye(n, m, k)?.into())
    }

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        U: Cast<f64> + Cast<T> + Copy,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T,
    ) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + num::Float + NormalOut<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self::Output, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T> + TypeCommon,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(_Tensor::tril(self.inner.as_ref(), k)?.into())
    }

    fn triu(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T> + TypeCommon,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(_Tensor::triu(self.inner.as_ref(), k)?.into())
    }

    fn identity(n: usize) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::identity(n)?.into())
    }

    fn from_owned<S: Into<Shape>>(data: &mut [T], shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::from_owned(data, shape)?.into())
    }
}

impl<T: CommonBounds, const DEVICE: usize> TensorCreator<T> for DiffTensor<T, Cpu, DEVICE> {
    type Output = DiffTensor<T, Cpu, DEVICE>;

    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        let ret = Tensor::<T, Cpu, DEVICE>::empty(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        let ret = Tensor::<T, Cpu, DEVICE>::zeros(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        let ret = Tensor::<T, Cpu, DEVICE>::ones(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn empty_like(&self) -> Result<Self::Output, TensorError> {
        let ret = self.inner.empty_like()?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zeros_like(&self) -> Result<Self::Output, TensorError> {
        let ret = self.inner.zeros_like()?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn ones_like(&self) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        let ret = self.inner.ones_like()?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self::Output, TensorError> {
        let ret = Tensor::full(val, shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn full_like(&self, val: T) -> Result<Self::Output, TensorError> {
        let ret = self.inner.full_like(val)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn arange<U>(start: U, end: U) -> Result<Self::Output, TensorError>
    where
        usize: Cast<T>,
        U: Cast<i64> + Cast<T> + Copy,
    {
        let ret = Tensor::arange(start, end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + Cast<usize>,
        usize: Cast<T>,
    {
        let ret = Tensor::arange_step(start, end, step)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError> {
        let ret = Tensor::<T, Cpu, DEVICE>::eye(n, m, k)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        U: Cast<f64> + Cast<T> + Copy,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let ret = Tensor::linspace(start, end, num, include_end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T,
    ) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + num::Float + NormalOut<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let ret = Tensor::logspace(start, end, num, include_end, base)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self::Output, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64>,
    {
        let ret = Tensor::geomspace(start, end, n, include_end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        let ret = Tensor::tri(n, m, k, low_triangle)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn tril(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T> + TypeCommon,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        let ret = self.inner.tril(k)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| unimplemented!())),
        })
    }

    fn triu(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T> + TypeCommon,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        let ret = self.inner.triu(k)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| unimplemented!())),
        })
    }

    fn identity(n: usize) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        let ret = Tensor::identity(n)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn from_owned<S: Into<Shape>>(data: &mut [T], shape: S) -> Result<Self::Output, TensorError> {
        let ret = Tensor::from_owned(data, shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }
}
