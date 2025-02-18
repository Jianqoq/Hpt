use std::borrow::BorrowMut;
use std::ops::BitAnd;

use crate::ops::cpu::tensor_internal::float_out_unary::FloatBinaryType;
use crate::ops::cuda::cuda_utils::get_module_name_1;
use crate::ops::cuda::utils::reduce::reduce::{reduce, reduce2};
use crate::ops::cuda::utils::unary::unary::uary_fn_with_out_simd;
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use hpt_common::axis::axis::{process_axes, Axis};
use hpt_common::error::base::TensorError;
use hpt_cudakernels::{ALL, ANY, MAX, MIN, NANPROD, NANSUM, PROD, REDUCEL1, SUM};
use hpt_traits::{
    CommonBounds, EvalReduce, FloatReduce, NormalEvalReduce, NormalReduce, TensorInfo,
};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use hpt_types::traits::SimdSelect;
use hpt_types::type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut};

impl<T: CommonBounds + DeviceRepr + CudaType + Cast<f64>, const DEVICE_ID: usize> NormalReduce<T>
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
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType + Cast<f64>,
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
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType + Cast<f64>,
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

impl<T, const DEVICE: usize> FloatReduce<T> for _Tensor<T, Cuda, DEVICE>
where
    T: FloatOutBinary + CommonBounds + Cast<FloatBinaryType<T>> + DeviceRepr + CudaType + Cast<f64>,
    FloatBinaryType<T>: CommonBounds + FloatOutUnary<Output = FloatBinaryType<T>>,
    f64: Cast<FloatBinaryType<T>>,
    FloatBinaryType<T>: NormalOut<T, Output = FloatBinaryType<T>>
        + NormalOut<<T as FloatOutUnary>::Output, Output = FloatBinaryType<T>>
        + DeviceRepr
        + CudaType,
    Scalar<T>: FloatOutBinary<Output = Scalar<FloatBinaryType<T>>>,
{
    type Output = _Tensor<FloatBinaryType<T>, Cuda, DEVICE>;

    #[track_caller]
    fn mean<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>, Cuda, DEVICE>, TensorError> {
        unimplemented!()
    }
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        _: S,
        _: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>, Cuda, DEVICE>, TensorError> {
        unimplemented!()
    }
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        _: S,
        _: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>, Cuda, DEVICE>, TensorError> {
        unimplemented!()
    }
    #[track_caller]
    fn logsumexp<S: Into<Axis>>(
        &self,
        _: S,
        _: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>, Cuda, DEVICE>, TensorError>
    where
        T: CommonBounds,
    {
        unimplemented!()
    }
}
