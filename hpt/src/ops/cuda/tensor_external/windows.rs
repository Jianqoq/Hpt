use crate::{ops::cpu::tensor_internal::windows::Simd, tensor::Tensor, tensor_base::_Tensor, Cuda};
use cudarc::driver::DeviceRepr;
use hpt_traits::CommonBounds;
use hpt_types::{
    cast::Cast,
    dtype::FloatConst,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};
use std::ops::{Mul, Sub};

type FBO<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID>
where
    f64: Cast<FBO<T>>,
    T: CommonBounds + FloatOutBinary + DeviceRepr,
    FBO<T>: CommonBounds
        + FloatOutUnary<Output = FBO<T>>
        + Mul<Output = FBO<T>>
        + Sub<Output = FBO<T>>
        + FloatConst
        + DeviceRepr,
    FBO<T>: std::ops::Neg<Output = FBO<T>>,
    FBO<T>: NormalOut<FBO<T>, Output = FBO<T>> + FloatOutBinary<FBO<T>, Output = FBO<T>>,
    Simd<T>: NormalOut<Simd<T>, Output = Simd<T>>
        + FloatOutBinary<Simd<T>, Output = Simd<T>>
        + FloatOutUnary<Output = Simd<T>>,
    usize: Cast<FBO<T>>,
    i64: Cast<T>,
{
    /// Generates a Hamming window of a specified length.
    ///
    /// The `hamming_window` function creates a Hamming window, which is commonly used in signal processing for smoothing data or reducing spectral leakage.
    ///
    /// The Hamming window is mathematically defined as:
    ///
    /// ```text
    /// w(n) = 0.54 - 0.46 * cos(2πn / (N - 1))
    /// ```
    ///
    /// where `N` is the `window_length` and `n` ranges from 0 to `N-1`.
    ///
    /// # Parameters
    ///
    /// - `window_length`: The length of the window.
    /// - `periodic`: If `true`, creates a periodic window, suitable for use in spectral analysis.
    ///
    /// # Returns
    ///
    /// - A tensor containing the Hamming window.
    #[track_caller]
    pub fn hamming_window(
        window_length: i64,
        periodic: bool,
    ) -> anyhow::Result<Tensor<FBO<T>, Cuda, DEVICE_ID>> {
        Ok(Tensor::from(
            _Tensor::<T, Cuda, DEVICE_ID>::hamming_window(window_length, periodic)?.into(),
        ))
    }

    /// Generates a Hann window of a specified length.
    ///
    /// The `hann_window` function creates a Hann window, which is used in signal processing to taper data and reduce spectral leakage.
    ///
    /// The Hann window is mathematically defined as:
    ///
    /// ```text
    /// w(n) = 0.5 * (1 - cos(2πn / (N - 1)))
    /// ```
    ///
    /// where `N` is the `window_length` and `n` ranges from 0 to `N-1`.
    ///
    /// # Parameters
    ///
    /// - `window_length`: The length of the window.
    /// - `periodic`: If `true`, creates a periodic window, suitable for use in spectral analysis.
    ///
    /// # Returns
    ///
    /// - A tensor containing the Hann window.
    #[track_caller]
    pub fn hann_window(
        window_length: i64,
        periodic: bool,
    ) -> anyhow::Result<Tensor<FBO<T>, Cuda, DEVICE_ID>> {
        Ok(Tensor::from(
            _Tensor::<T, Cuda, DEVICE_ID>::hann_window(window_length, periodic)?.into(),
        ))
    }

    /// Generates a Blackman window tensor.
    ///
    /// A Blackman window is commonly used in signal processing to reduce spectral leakage.
    /// This method generates a tensor representing the Blackman window, which can be used
    /// for tasks like filtering or analysis in the frequency domain. The window can be
    /// either periodic or symmetric, depending on the `periodic` parameter.
    ///
    /// # Arguments
    ///
    /// * `window_length` - The length of the window, specified as an `i64`. This determines
    ///   the number of elements in the output tensor.
    /// * `periodic` - A boolean flag indicating whether the window should be periodic or symmetric:
    ///   - If `true`, the window will be periodic, which is typically used for spectral analysis.
    ///   - If `false`, the window will be symmetric, which is typically used for filtering.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `<T as FloatOutBinary>::Output`
    #[track_caller]
    pub fn blackman_window(
        window_length: i64,
        periodic: bool,
    ) -> anyhow::Result<Tensor<FBO<T>, Cuda, DEVICE_ID>>
    where
        T: FloatConst,
        i64: Cast<FBO<T>>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::blackman_window(window_length, periodic)?.into())
    }
}
