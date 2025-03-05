use crate::{tensor::Tensor, tensor_base::_Tensor, Cpu};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, WindowOps};

impl<T, const DEVICE: usize, Al> WindowOps for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    _Tensor<T, Cpu, DEVICE, Al>: WindowOps<Output = _Tensor<T, Cpu, DEVICE, Al>>,
    _Tensor<T, Cpu, DEVICE, Al>: Into<Tensor<T, Cpu, DEVICE, Al>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;
    type Meta = T;
    #[track_caller]
    fn hamming_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::hamming_window(window_length, periodic)?.into())
    }
    #[track_caller]
    fn hann_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::hann_window(window_length, periodic)?.into())
    }
    #[track_caller]
    fn blackman_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::blackman_window(window_length, periodic)?.into())
    }
}
