use std::{borrow::BorrowMut, ops::BitAnd};

use crate::ops::cpu::tensor_internal::float_out_unary::FloatBinaryType;
use crate::tensor::Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, EvalReduce, FloatReduce, NormalEvalReduce, NormalReduce};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::type_promote::{FloatOutBinary, FloatOutUnary, NormalOut};
use hpt_types::{into_scalar::Cast, traits::SimdSelect, type_promote::Eval};

impl<T: CommonBounds + DeviceRepr + CudaType + Cast<f64>, const DEVICE_ID: usize> NormalReduce<T>
    for Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Self;

    fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.sum(axes, keep_dims)?.into())
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        mut out: O,
    ) -> Result<Self::Output, TensorError>
    where
        O: BorrowMut<Self::Output>,
    {
        Ok(self
            .inner
            .sum_(axes, keep_dims, init_out, out.borrow_mut())?
            .into())
    }

    // fn sum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.sum_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.prod(axis, keep_dims)?.into())
    }

    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.prod_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.min(axis, keep_dims)?.into())
    }

    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.min_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.max(axis, keep_dims)?.into())
    }

    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.max_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.reducel1(axis, keep_dims)?.into())
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.sum_square(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE_ID: usize> EvalReduce for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType + Cast<f64>,
{
    type BoolOutput = Tensor<bool, Cuda, DEVICE_ID>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::BoolOutput, TensorError> {
        Ok(self.inner.all(axis, keep_dims)?.into())
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::BoolOutput, TensorError> {
        Ok(self.inner.any(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE_ID: usize> NormalEvalReduce<T> for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType + Cast<f64>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec> + Copy,
    <T::Vec as Eval>::Output: BitAnd<Output = <T::Vec as Eval>::Output>,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.nansum(axes, keep_dims)?.into())
    }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.nanprod(axis, keep_dims)?.into())
    }

    fn nansum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        mut out: O,
    ) -> Result<Self::Output, TensorError>
    where
        O: BorrowMut<Self::Output>,
    {
        Ok(self
            .inner
            .nansum_(axes, keep_dims, init_out, out.borrow_mut())?
            .into())
    }
}

impl<T, const DEVICE: usize> FloatReduce<T> for Tensor<T, Cuda, DEVICE>
where
    T: FloatOutBinary + CommonBounds + Cast<FloatBinaryType<T>> + DeviceRepr + CudaType + Cast<f64>,
    FloatBinaryType<T>: CommonBounds + FloatOutUnary<Output = FloatBinaryType<T>>,
    f64: Cast<FloatBinaryType<T>>,
    FloatBinaryType<T>: NormalOut<T, Output = FloatBinaryType<T>>
        + NormalOut<<T as FloatOutUnary>::Output, Output = FloatBinaryType<T>>
        + DeviceRepr
        + CudaType,
    Scalar<FloatBinaryType<T>>: FloatOutBinary<Output = Scalar<FloatBinaryType<T>>>
        + FloatOutUnary<Output = Scalar<FloatBinaryType<T>>>
        + NormalOut<Output = Scalar<FloatBinaryType<T>>>,
{
    type Output = Tensor<FloatBinaryType<T>, Cuda, DEVICE>;

    #[track_caller]
    fn mean<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.mean(axes, keep_dims)?.into())
    }
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.reducel2(axes, keep_dims)?.into())
    }
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.reducel3(axes, keep_dims)?.into())
    }
    #[track_caller]
    fn logsumexp<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        T: CommonBounds,
    {
        Ok(self.inner.logsumexp(axes, keep_dims)?.into())
    }
}
