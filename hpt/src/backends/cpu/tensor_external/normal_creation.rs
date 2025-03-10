use std::{cell::RefCell, rc::Rc};

use crate::{
    tensor::{DiffTensor, Tensor},
    tensor_base::_Tensor,
    BoolVector,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, NormalOut},
};

impl<T: CommonBounds, const DEVICE: usize, Al> TensorCreator for Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;
    type Meta = T;

    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::ones(shape)?.into())
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
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::full_like(self.inner.as_ref(), val)?.into())
    }

    fn arange<U>(start: U, end: U) -> Result<Self::Output, TensorError>
    where
        usize: Cast<T>,
        U: Cast<i64> + Cast<T> + Copy,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + Cast<f64>,
        f64: Cast<T>,
        usize: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::eye(n, m, k)?.into())
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
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace<V: Cast<T>>(
        start: V,
        end: V,
        num: usize,
        include_end: bool,
        base: V,
    ) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + num::Float + FloatOutBinary<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace<V: Cast<T>>(
        start: V,
        end: V,
        n: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64> + FloatOutBinary<T, Output = T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::tri(n, m, k, low_triangle)?.into())
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
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::identity(n)?.into())
    }
}

impl<T: CommonBounds, const DEVICE: usize, Al> TensorCreator for DiffTensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = DiffTensor<T, Cpu, DEVICE, Al>;
    type Meta = T;

    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        let ret = Tensor::<T, Cpu, DEVICE, Al>::empty(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        let ret = Tensor::<T, Cpu, DEVICE, Al>::zeros(shape)?;
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
        let ret = Tensor::<T, Cpu, DEVICE, Al>::ones(shape)?;
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
        let ret = Tensor::<T, Cpu, DEVICE, Al>::full(val, shape)?;
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
        let ret = Tensor::<T, Cpu, DEVICE, Al>::arange(start, end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + Cast<f64>,
        f64: Cast<T>,
        usize: Cast<T>,
    {
        let ret = Tensor::<T, Cpu, DEVICE, Al>::arange_step(start, end, step)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError> {
        let ret = Tensor::<T, Cpu, DEVICE, Al>::eye(n, m, k)?;
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
        let ret = Tensor::<T, Cpu, DEVICE, Al>::linspace(start, end, num, include_end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn logspace<V: Cast<T>>(
        start: V,
        end: V,
        num: usize,
        include_end: bool,
        base: V,
    ) -> Result<Self::Output, TensorError>
    where
        T: Cast<f64> + num::Float + FloatOutBinary<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let ret = Tensor::<T, Cpu, DEVICE, Al>::logspace(start, end, num, include_end, base)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn geomspace<V: Cast<T>>(
        start: V,
        end: V,
        n: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64> + FloatOutBinary<T, Output = T>,
    {
        let ret = Tensor::<T, Cpu, DEVICE, Al>::geomspace(start, end, n, include_end)?;
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
        let ret = Tensor::<T, Cpu, DEVICE, Al>::tri(n, m, k, low_triangle)?;
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
        let ret = Tensor::<T, Cpu, DEVICE, Al>::identity(n)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }
}
