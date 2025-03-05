use crate::{tensor::Tensor, tensor_base::_Tensor, Cuda};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, WindowOps};

use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};

impl<T, const DEVICE_ID: usize, Al> WindowOps for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds,
    _Tensor<T, Cuda, DEVICE_ID, Al>: WindowOps<Output = _Tensor<T, Cuda, DEVICE_ID, Al>>,
    _Tensor<T, Cuda, DEVICE_ID, Al>: Into<Tensor<T, Cuda, DEVICE_ID, Al>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cuda, DEVICE_ID, Al>;
    type Meta = T;
    #[track_caller]
    fn hamming_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::hamming_window(window_length, periodic)?.into())
    }

    #[track_caller]
    fn hann_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::hann_window(window_length, periodic)?.into())
    }

    #[track_caller]
    fn blackman_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::blackman_window(window_length, periodic)?.into())
    }
}
