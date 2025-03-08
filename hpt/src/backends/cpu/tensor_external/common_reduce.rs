use crate::tensor::Tensor;
use crate::tensor_base::_Tensor;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_traits::{
    ops::reduce::{EvalReduce, FloatReduce, NormalEvalReduce, NormalReduce},
    tensor::CommonBounds,
};
use hpt_types::{
    into_scalar::Cast,
    type_promote::{Eval, FloatOutBinary},
    vectors::traits::SimdSelect,
};
use std::borrow::BorrowMut;

type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T: CommonBounds, const DEVICE: usize, Al> NormalReduce<T> for Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator + Send + Sync + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.sum(axes, keep_dims)?.into())
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        mut out: O,
    ) -> std::result::Result<Self::Output, TensorError>
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
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.sum_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.prod(axis, keep_dims)?.into())
    }

    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.prod_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        Ok(self.inner.min(axis, keep_dims)?.into())
    }

    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.min_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        Ok(self.inner.max(axis, keep_dims)?.into())
    }

    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.max_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.reducel1(axis, keep_dims)?.into())
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.sum_square(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE: usize, Al> EvalReduce for Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator + Send + Sync + 'static,
    Al::Output: AllocatorOutputRetrive,
    T: CommonBounds,
    _Tensor<T, Cpu, DEVICE, Al>: EvalReduce<BoolOutput = _Tensor<bool, Cpu, DEVICE, Al>>,
{
    type BoolOutput = Tensor<bool, Cpu, DEVICE, Al>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        Ok(self.inner.all(axis, keep_dims)?.into())
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        Ok(self.inner.any(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE: usize, Al> NormalEvalReduce<T> for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    Al: Allocator + Send + Sync + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.nansum(axes, keep_dims)?.into())
    }

    fn nansum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        mut out: O,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        O: BorrowMut<Self::Output>,
    {
        Ok(self
            .inner
            .nansum_(axes, keep_dims, init_out, out.borrow_mut())?
            .into())
    }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.nanprod(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE: usize, Al> FloatReduce<T> for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    _Tensor<T, Cpu, DEVICE, Al>:
        FloatReduce<T, Output = _Tensor<FloatBinaryType<T>, Cpu, DEVICE, Al>>,
    Al: Allocator + Send + Sync + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatBinaryType<T>, Cpu, DEVICE, Al>;

    #[track_caller]
    fn mean<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.mean(axis, keep_dims)?.into())
    }

    #[allow(unused)]
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.reducel2(axis, keep_dims)?.into())
    }

    #[allow(unused)]
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.reducel3(axis, keep_dims)?.into())
    }

    #[allow(unused)]
    #[track_caller]
    fn logsumexp<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.logsumexp(axis, keep_dims)?.into())
    }
}
