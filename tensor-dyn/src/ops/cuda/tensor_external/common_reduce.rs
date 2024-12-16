use std::borrow::Borrow;

use crate::tensor::Tensor;
use crate::Cuda;
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use tensor_common::axis::Axis;
use tensor_traits::{CommonBounds, NormalReduce};

impl<T: CommonBounds + DeviceRepr + CudaTypeName, const DEVICE_ID: usize> NormalReduce<T>
    for Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Self;

    fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.sum(axes, keep_dims)?.into())
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> anyhow::Result<Self::Output>
    where
        O: Borrow<Self::Output>,
    {
        Ok(self
            .inner
            .sum_(axes, keep_dims, init_out, out.borrow())?
            .into())
    }

    fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        Ok(self.inner.sum_with_init(init_val, axes, keep_dims)?.into())
    }

    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.prod(axis, keep_dims)?.into())
    }

    fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        Ok(self.inner.prod_with_init(init_val, axes, keep_dims)?.into())
    }

    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        Ok(self.inner.min(axis, keep_dims)?.into())
    }

    fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self> {
        Ok(self.inner.min_with_init(init_val, axes, keep_dims)?.into())
    }

    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        Ok(self.inner.max(axis, keep_dims)?.into())
    }

    fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self> {
        Ok(self.inner.max_with_init(init_val, axes, keep_dims)?.into())
    }

    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.reducel1(axis, keep_dims)?.into())
    }

    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.sum_square(axis, keep_dims)?.into())
    }
}
