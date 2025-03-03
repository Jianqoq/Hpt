use crate::{ops::cpu::tensor_internal::windows::Simd, tensor::Tensor, tensor_base::_Tensor, Cuda};
use cudarc::driver::DeviceRepr;
use hpt_common::error::base::TensorError;
use hpt_traits::CommonBounds;
use hpt_types::dtype::CudaType;
use hpt_types::{
    dtype::FloatConst,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};
use std::ops::{Mul, Sub};

use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
type FBO<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE_ID: usize, Al> Tensor<T, Cuda, DEVICE_ID, Al>
where
    f64: Cast<FBO<T>>,
    T: CommonBounds + FloatOutBinary + DeviceRepr + CudaType,
    FBO<T>: CommonBounds
        + FloatOutUnary<Output = FBO<T>>
        + Mul<Output = FBO<T>>
        + Sub<Output = FBO<T>>
        + FloatConst
        + DeviceRepr
        + CudaType,
    FBO<T>: std::ops::Neg<Output = FBO<T>>,
    FBO<T>: NormalOut<FBO<T>, Output = FBO<T>> + FloatOutBinary<FBO<T>, Output = FBO<T>>,
    Simd<T>: NormalOut<Simd<T>, Output = Simd<T>>
        + FloatOutBinary<Simd<T>, Output = Simd<T>>
        + FloatOutUnary<Output = Simd<T>>,
    usize: Cast<FBO<T>>,
    i64: Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    /// Computes the Hamming window for a given window length.
    ///
    /// This method generates a Hamming window of a specified length. The Hamming window is a type of window function that
    /// is commonly used in signal processing and spectral analysis. It is named after Julius von Hann, who first proposed
    /// the Hann window, and Richard Hamming, who later derived the formula for the window.
    ///
    #[track_caller]
    pub fn hamming_window(
        window_length: i64,
        periodic: bool,
    ) -> Result<Tensor<FBO<T>, Cuda, DEVICE_ID, Al>, TensorError> {
        Ok(Tensor::from(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::hamming_window(window_length, periodic)?.into(),
        ))
    }

    /// Computes the Hann window for a given window length.
    ///
    /// This method generates a Hann window of a specified length. The Hann window is a type of window function that
    /// is commonly used in signal processing and spectral analysis. It is named after Julius von Hann, who first proposed
    /// the Hann window, and Richard Hamming, who later derived the formula for the window.
    #[track_caller]
    pub fn hann_window(
        window_length: i64,
        periodic: bool,
    ) -> Result<Tensor<FBO<T>, Cuda, DEVICE_ID, Al>, TensorError> {
        Ok(Tensor::from(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::hann_window(window_length, periodic)?.into(),
        ))
    }

    /// Computes the Blackman window for a given window length.
    ///
    /// This method generates a Blackman window of a specified length. The Blackman window is a type of window function that
    /// is commonly used in signal processing and spectral analysis. It is named after Julius von Hann, who first proposed
    /// the Hann window, and Richard Hamming, who later derived the formula for the window.
    #[track_caller]
    pub fn blackman_window(
        window_length: i64,
        periodic: bool,
    ) -> Result<Tensor<FBO<T>, Cuda, DEVICE_ID, Al>, TensorError>
    where
        T: FloatConst,
        i64: Cast<FBO<T>>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::blackman_window(window_length, periodic)?.into())
    }
}
