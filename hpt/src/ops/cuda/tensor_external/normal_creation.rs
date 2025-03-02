use crate::{tensor_base::_Tensor, BoolVector, Cuda, Tensor};
use cudarc::driver::DeviceRepr;
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};
use hpt_types::{dtype::CudaType, into_scalar::Cast, type_promote::NormalOut};

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize> TensorCreator<T>
    for Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Tensor<T, Cuda, DEVICE_ID>;

    fn empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> std::result::Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::ones(shape)?.into())
    }

    fn empty_like(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::empty(self.inner.as_ref().shape())?.into())
    }

    fn zeros_like(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.as_ref().zeros_like()?.into())
    }

    fn ones_like(&self) -> std::result::Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(self.inner.as_ref().ones_like()?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.as_ref().full_like(val)?.into())
    }

    fn arange<U>(start: U, end: U) -> std::result::Result<Self::Output, TensorError>
    where
        usize: Cast<T>,
        U: Cast<i64> + Cast<T> + Copy,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> std::result::Result<Self::Output, TensorError>
    where
        T: Cast<f64> + Cast<usize>,
        usize: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::eye(n, m, k)?.into())
    }

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Cast<f64> + Cast<T> + Copy,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        T: Cast<f64> + num::Float + NormalOut<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(
        start: T,
        end: T,
        n: usize,
        include_end: bool,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(
        n: usize,
        m: usize,
        k: i64,
        low_triangle: bool,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> std::result::Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(self.inner.as_ref().tril(k)?.into())
    }

    fn triu(&self, k: i64) -> std::result::Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(self.inner.as_ref().triu(k)?.into())
    }

    fn identity(n: usize) -> std::result::Result<Self::Output, TensorError>
    where
        u8: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::eye(n, n, 0)?.into())
    }

    fn from_owned<S: Into<Shape>>(data: &mut [T], shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::from_owned(data, shape)?.into())
    }
}
