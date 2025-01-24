use std::{ cell::RefCell, rc::Rc };

use crate::{ backend::Cpu, tensor::{ DiffTensor, Tensor }, tensor_base::_Tensor, BoolVector };
use tensor_common::{ error::base::TensorError, shape::shape::Shape };
use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::NormalOut,
};

impl<T: CommonBounds, const DEVICE: usize> TensorCreator<T> for Tensor<T, Cpu, DEVICE> {
    type Output = Tensor<T, Cpu, DEVICE>;

    fn empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError>
        where u8: IntoScalar<T>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::ones(shape)?.into())
    }

    fn empty_like(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::empty_like(self.inner.as_ref())?.into())
    }

    fn zeros_like(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::zeros_like(self.inner.as_ref())?.into())
    }

    fn ones_like(&self) -> std::result::Result<Self::Output, TensorError> where u8: IntoScalar<T> {
        Ok(_Tensor::ones_like(self.inner.as_ref())?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::full_like(self.inner.as_ref(), val)?.into())
    }

    fn arange<U>(start: U, end: U) -> std::result::Result<Self::Output, TensorError>
        where T: FromScalar<U>, usize: IntoScalar<T>, U: Convertor + IntoScalar<T> + Copy
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> std::result::Result<Self::Output, TensorError>
        where T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> std::result::Result<Self::Output, TensorError>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::eye(n, m, k)?.into())
    }

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool
    )
        -> std::result::Result<Self::Output, TensorError>
        where
            T: Convertor + NormalOut<T, Output = T>,
            U: Convertor + IntoScalar<T> + Copy,
            usize: IntoScalar<T>,
            f64: IntoScalar<T>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T
    ) -> std::result::Result<Self::Output, TensorError>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(
        start: T,
        end: T,
        n: usize,
        include_end: bool
    )
        -> std::result::Result<Self::Output, TensorError>
        where f64: IntoScalar<T>, usize: IntoScalar<T>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(
        n: usize,
        m: usize,
        k: i64,
        low_triangle: bool
    ) -> std::result::Result<Self::Output, TensorError>
        where u8: IntoScalar<T>
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> std::result::Result<Self::Output, TensorError>
        where
            T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
            T::Vec: NormalOut<BoolVector, Output = T::Vec>
    {
        Ok(_Tensor::tril(self.inner.as_ref(), k)?.into())
    }

    fn triu(&self, k: i64) -> std::result::Result<Self::Output, TensorError>
        where
            T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
            T::Vec: NormalOut<BoolVector, Output = T::Vec>
    {
        Ok(_Tensor::triu(self.inner.as_ref(), k)?.into())
    }

    fn identity(n: usize) -> std::result::Result<Self::Output, TensorError> where u8: IntoScalar<T> {
        Ok(_Tensor::<T, Cpu, DEVICE>::identity(n)?.into())
    }
}

impl<T: CommonBounds, const DEVICE: usize> TensorCreator<T> for DiffTensor<T, Cpu, DEVICE> {
    type Output = DiffTensor<T, Cpu, DEVICE>;

    fn empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError> {
        let ret = Tensor::empty(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError> {
        let ret = Tensor::zeros(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn ones<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError>
        where u8: IntoScalar<T>
    {
        let ret = Tensor::ones(shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn empty_like(&self) -> std::result::Result<Self::Output, TensorError> {
        let ret = self.inner.empty_like()?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zeros_like(&self) -> std::result::Result<Self::Output, TensorError> {
        let ret = self.inner.zeros_like()?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn ones_like(&self) -> std::result::Result<Self::Output, TensorError> where u8: IntoScalar<T> {
        let ret = self.inner.ones_like()?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> std::result::Result<Self::Output, TensorError> {
        let ret = Tensor::full(val, shape)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn full_like(&self, val: T) -> std::result::Result<Self::Output, TensorError> {
        let ret = self.inner.full_like(val)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn arange<U>(start: U, end: U) -> std::result::Result<Self::Output, TensorError>
        where T: FromScalar<U>, usize: IntoScalar<T>, U: Convertor + IntoScalar<T> + Copy
    {
        let ret = Tensor::arange(start, end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn arange_step(start: T, end: T, step: T) -> std::result::Result<Self::Output, TensorError>
        where T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>
    {
        let ret = Tensor::arange_step(start, end, step)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn eye(n: usize, m: usize, k: usize) -> std::result::Result<Self::Output, TensorError>
    {
        let ret = Tensor::eye(n, m, k)?;
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
        include_end: bool
    )
        -> std::result::Result<Self::Output, TensorError>
        where
            T: Convertor + NormalOut<T, Output = T>,
            U: Convertor + IntoScalar<T> + Copy,
            usize: IntoScalar<T>,
            f64: IntoScalar<T>
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
        base: T
    ) -> std::result::Result<Self::Output, TensorError>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        let ret = Tensor::logspace(start, end, num, include_end, base)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn geomspace(
        start: T,
        end: T,
        n: usize,
        include_end: bool
    )
        -> std::result::Result<Self::Output, TensorError>
        where f64: IntoScalar<T>, usize: IntoScalar<T>
    {
        let ret = Tensor::geomspace(start, end, n, include_end)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn tri(
        n: usize,
        m: usize,
        k: i64,
        low_triangle: bool
    ) -> std::result::Result<Self::Output, TensorError>
        where u8: IntoScalar<T>
    {
        let ret = Tensor::tri(n, m, k, low_triangle)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn tril(&self, k: i64) -> std::result::Result<Self::Output, TensorError>
        where
            T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
            T::Vec: NormalOut<BoolVector, Output = T::Vec>
    {
        let ret = self.inner.tril(k)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| unimplemented!())),
        })
    }

    fn triu(&self, k: i64) -> std::result::Result<Self::Output, TensorError>
        where
            T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
            T::Vec: NormalOut<BoolVector, Output = T::Vec>
    {
        let ret = self.inner.triu(k)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| unimplemented!())),
        })
    }

    fn identity(n: usize) -> std::result::Result<Self::Output, TensorError> where u8: IntoScalar<T> {
        let ret = Tensor::identity(n)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }
}
