use std::borrow::Borrow;

use crate::ops::cuda::reduce::{reduce, reduce2};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use cudarc::types::CudaTypeName;
use tensor_common::axis::{process_axes, Axis};
use tensor_traits::{CommonBounds, NormalReduce, TensorInfo};
use tensor_types::type_promote::NormalOut;
use tensor_types::type_promote::NormalOutUnary;

impl<T: CommonBounds + DeviceRepr + CudaTypeName, const DEVICE_ID: usize> NormalReduce<T>
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Self;

    fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
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
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out.borrow().clone()),
        )
    }

    fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None,
        )
    }

    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            None,
        )
    }

    fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None,
        )
    }

    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            T::INF,
            keep_dims,
            false,
            None,
        )
    }

    fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None,
        )
    }

    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            None,
        )
    }

    fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None,
        )
    }

    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| {
                let b_abs = b._abs();
                a._add(b_abs)
            },
            |a, b| {
                let b_abs = b._abs();
                a._add(b_abs)
            },
            |a, b| a._add(b._abs()),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
    }

    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(
            self,
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            |a, b| a._add(b),
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
    }
}
