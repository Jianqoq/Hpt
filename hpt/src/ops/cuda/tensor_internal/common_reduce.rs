use std::borrow::BorrowMut;
use std::ops::BitAnd;

use crate::ops::cuda::utils::reduce::reduce::{reduce, reduce2};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use hpt_common::axis::axis::{process_axes, Axis};
use hpt_common::error::base::TensorError;
use hpt_cudakernels::{ALL, ANY, MAX, MIN, NANPROD, NANSUM, PROD, REDUCEL1, SUM};
use hpt_traits::{CommonBounds, EvalReduce, NormalEvalReduce, NormalReduce, TensorInfo};
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use hpt_types::traits::SimdSelect;
use hpt_types::type_promote::Eval;

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize> NormalReduce<T>
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(self, &axes, T::ZERO, keep_dims, false, &SUM, "sum", None)
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        O: BorrowMut<Self::Output>,
    {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            &SUM,
            "sum",
            Some(out.borrow().clone()),
        )
    }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axis, self.ndim())?;
        reduce(self, &axes, T::ONE, keep_dims, false, &PROD, "prod", None)
    }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(self, &axes, T::INF, keep_dims, false, &MIN, "min", None)
    }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(self, &axes, T::NEG_INF, keep_dims, false, &MAX, "max", None)
    }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &REDUCEL1,
            "reducel1",
            None,
        )
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(self, &axes, T::ZERO, keep_dims, false, &SUM, "sum", None)
    }
}

impl<T, const DEVICE_ID: usize> EvalReduce for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType,
{
    type BoolOutput = _Tensor<bool, Cuda, DEVICE_ID>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(self, &axes, true, keep_dims, false, &ALL, "all", None)
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(self, &axes, false, keep_dims, false, &ANY, "any", None)
    }
}

impl<T, const DEVICE_ID: usize> NormalEvalReduce<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec> + Copy,
    <T::Vec as Eval>::Output: BitAnd<Output = <T::Vec as Eval>::Output>,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &NANSUM,
            "nansum",
            None,
        )
    }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            &axes,
            T::ONE,
            keep_dims,
            false,
            &NANPROD,
            "nanprod",
            None,
        )
    }

    fn nansum_<S: Into<hpt_common::axis::axis::Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        mut out: O,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError>
    where
        O: BorrowMut<Self::Output>,
    {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            &NANSUM,
            "nansum",
            Some(out.borrow_mut().clone()),
        )
    }
}
