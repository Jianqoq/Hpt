use crate::{tensor_base::_Tensor, BoolVector, Cuda, Tensor};
use anyhow::Result;
use cudarc::driver::DeviceRepr;
use tensor_common::shape::Shape;
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    into_scalar::IntoScalar,
    type_promote::NormalOut,
};

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> TensorCreator<T>
    for Tensor<T, Cuda, DEVICE_ID>
{
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::ones(shape)?.into())
    }

    fn empty_like(&self) -> Result<Self> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::empty(self.inner.as_ref().shape())?.into())
    }

    fn zeros_like(&self) -> Result<Self> {
        Ok(self.inner.as_ref().zeros_like()?.into())
    }

    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(self.inner.as_ref().ones_like()?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> Result<Self> {
        Ok(self.inner.as_ref().full_like(val)?.into())
    }

    fn arange<U>(start: U, end: U) -> Result<Self>
    where
        T: Convertor + FromScalar<U>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self>
    where
        T: Convertor + FromScalar<usize>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::eye(n, m, k)?.into())
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        f64: IntoScalar<T>,
        usize: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(self.inner.as_ref().tril(k)?.into())
    }

    fn triu(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(self.inner.as_ref().triu(k)?.into())
    }

    fn identity(n: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::eye(n, n, 0)?.into())
    }
}
