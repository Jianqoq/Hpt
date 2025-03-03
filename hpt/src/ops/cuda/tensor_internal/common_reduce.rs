use std::borrow::BorrowMut;
use std::ops::BitAnd;

use crate::ops::cpu::tensor_internal::float_out_unary::FloatBinaryType;
use crate::ops::cuda::utils::reduce::reduce::{reduce, reduce2, reduce3};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::axis::axis::{process_axes, Axis};
use hpt_common::error::base::TensorError;
use hpt_traits::{
    CommonBounds, EvalReduce, FloatReduce, NormalEvalReduce, NormalReduce, TensorInfo,
};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::traits::SimdSelect;
use hpt_types::type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut};

impl<T: CommonBounds + DeviceRepr + CudaType + Cast<f64>, const DEVICE_ID: usize, Al>
    NormalReduce<T> for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::SUM,
            "reduce",
            "sum",
            None,
        )
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
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            &hpt_cudakernels::SUM,
            "reduce",
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
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ONE,
            keep_dims,
            false,
            &hpt_cudakernels::PROD,
            "reduce",
            "prod",
            None,
        )
    }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::INF,
            keep_dims,
            false,
            &hpt_cudakernels::MIN,
            "reduce",
            "min",
            None,
        )
    }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            &hpt_cudakernels::MAX,
            "reduce",
            "max",
            None,
        )
    }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::REDUCEL1,
            "reduce",
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
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::SUM_SQUARE,
            "reduce",
            "sumsquare",
            None,
        )
    }
}

impl<T, const DEVICE_ID: usize, Al> EvalReduce for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType + Cast<f64>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type BoolOutput = _Tensor<bool, Cuda, DEVICE_ID, Al>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2::<T, bool, bool, DEVICE_ID, Al>(
            self,
            &axes,
            true,
            keep_dims,
            false,
            &hpt_cudakernels::ALL,
            "reduce",
            "all",
            None,
        )
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2::<T, bool, bool, DEVICE_ID, Al>(
            self,
            &axes,
            false,
            keep_dims,
            false,
            &hpt_cudakernels::ANY,
            "reduce",
            "any",
            None,
        )
    }
}

impl<T, const DEVICE_ID: usize, Al> NormalEvalReduce<T> for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType + Cast<f64>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec> + Copy,
    <T::Vec as Eval>::Output: BitAnd<Output = <T::Vec as Eval>::Output>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::NANSUM,
            "reduce",
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
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ONE,
            keep_dims,
            false,
            &hpt_cudakernels::NANPROD,
            "reduce",
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
        reduce::<T, T, DEVICE_ID, Al>(
            self,
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            &hpt_cudakernels::NANSUM,
            "reduce",
            "nansum",
            Some(out.borrow_mut().clone()),
        )
    }
}

impl<T, const DEVICE: usize, Al> FloatReduce<T> for _Tensor<T, Cuda, DEVICE, Al>
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
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatBinaryType<T>, Cuda, DEVICE, Al>;

    #[track_caller]
    fn mean<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce3::<T, FloatBinaryType<T>, FloatBinaryType<T>, DEVICE, Al>(
            self,
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::MEAN,
            "reduce",
            "mean",
            None,
        )
    }
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce3::<T, FloatBinaryType<T>, FloatBinaryType<T>, DEVICE, Al>(
            self,
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::REDUCEL2,
            "reduce",
            "reducel2",
            None,
        )
    }
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce3::<T, FloatBinaryType<T>, FloatBinaryType<T>, DEVICE, Al>(
            self,
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::REDUCEL3,
            "reduce",
            "reducel3",
            None,
        )
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
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce3::<T, FloatBinaryType<T>, FloatBinaryType<T>, DEVICE, Al>(
            self,
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            &hpt_cudakernels::LOGSUMEXP,
            "reduce",
            "logsumexp",
            None,
        )
    }
}
