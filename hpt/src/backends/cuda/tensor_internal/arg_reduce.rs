use crate::{backend::Cuda, backends::cuda::utils::reduce::reduce::reduce3, tensor_base::_Tensor};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::{
    axis::axis::{process_axes, Axis},
    error::base::TensorError,
};
use hpt_traits::{
    ops::reduce::IndexReduce,
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::dtype::CudaType;
use hpt_types::{
    into_scalar::Cast,
    type_promote::{Cmp, NormalOut},
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ArgResult<T> {
    val: T,
    idx: i64,
}
unsafe impl<T: DeviceRepr> DeviceRepr for ArgResult<T> {}

impl<T, const DEVICE_ID: usize, Al> IndexReduce for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + NormalOut<Output = T> + Cmp + DeviceRepr + CudaType + Cast<i64>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<i64, Cuda, DEVICE_ID, Al>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce3::<T, i64, ArgResult<T>, DEVICE_ID, Al>(
            self,
            &axes,
            0,
            keep_dims,
            false,
            &hpt_cudakernels::ARGMAX,
            "reduce",
            "argmax",
            None,
        )
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce3::<T, i64, ArgResult<T>, DEVICE_ID, Al>(
            self,
            &axes,
            0,
            keep_dims,
            false,
            &hpt_cudakernels::ARGMIN,
            "reduce",
            "argmin",
            None,
        )
    }
}
