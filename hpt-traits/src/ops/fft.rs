use anyhow::Result;
use hpt_common::axis::axis::Axis;

/// A trait for Fast Fourier Transform (FFT) operations.
pub trait FFTOps
where
    Self: Sized,
{
    /// Computes the Fast Fourier Transform (FFT) of the tensor along a specified axis.
    ///
    /// The `fft` function computes the one-dimensional discrete Fourier Transform of the input tensor,
    /// converting the signal from the time domain to the frequency domain.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the FFT.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<ComplexType<T>>>`: A tensor of complex numbers representing the frequency components.
    ///
    /// # Notes
    ///
    /// - **Fourier Transform**: Converts time-domain signals to frequency-domain signals.
    /// - **Axis Specification**: The FFT is computed along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`ifft`]: Computes the inverse FFT of the tensor.
    /// - [`fft2`]: Computes the 2D FFT of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fft(&self, axis: i64) -> Result<Self>;

    /// Computes the inverse Fast Fourier Transform (IFFT) of the tensor along a specified axis.
    ///
    /// The `ifft` function computes the one-dimensional inverse discrete Fourier Transform of the input tensor,
    /// converting the signal from the frequency domain back to the time domain.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the IFFT.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatType<T>>>`: A tensor of real numbers representing the time-domain signal.
    ///
    /// # Notes
    ///
    /// - **Inverse Fourier Transform**: Converts frequency-domain signals back to the time domain.
    /// - **Axis Specification**: The IFFT is computed along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`fft`]: Computes the FFT of the tensor.
    /// - [`ifft2`]: Computes the 2D inverse FFT of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifft(&self, axis: i64) -> Result<Self>;

    /// Computes the 2D Fast Fourier Transform (FFT2) of the tensor.
    ///
    /// The `fft2` function computes the two-dimensional discrete Fourier Transform of the input tensor,
    /// converting the signal from the time domain to the frequency domain in two dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<ComplexType<T>>>`: A tensor of complex numbers representing the frequency components.
    ///
    /// # Notes
    ///
    /// - **Fourier Transform**: Converts 2D time-domain signals to frequency-domain signals.
    /// - **Multidimensional**: Operates over two axes at once.
    ///
    /// # See Also
    ///
    /// - [`ifft2`]: Computes the 2D inverse FFT of the tensor.
    /// - [`fft`]: Computes the 1D FFT of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fft2(&self, axis1: i64, axis2: i64) -> Result<Self>;

    /// Computes the 2D inverse Fast Fourier Transform (IFFT2) of the tensor.
    ///
    /// The `ifft2` function computes the two-dimensional inverse discrete Fourier Transform of the input tensor,
    /// converting the signal from the frequency domain back to the time domain in two dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatType<T>>>`: A tensor of real numbers representing the time-domain signal.
    ///
    /// # Notes
    ///
    /// - **Inverse Fourier Transform**: Converts 2D frequency-domain signals back to the time domain.
    /// - **Multidimensional**: Operates over two axes at once.
    ///
    /// # See Also
    ///
    /// - [`fft2`]: Computes the 2D FFT of the tensor.
    /// - [`ifft`]: Computes the 1D inverse FFT of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifft2(&self, axis1: i64, axis2: i64) -> Result<Self>;

    /// Computes the N-dimensional Fast Fourier Transform (FFTN) of the tensor.
    ///
    /// The `fftn` function computes the N-dimensional discrete Fourier Transform of the input tensor,
    /// converting the signal from the time domain to the frequency domain in N dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<ComplexType<T>>>`: A tensor of complex numbers representing the frequency components.
    ///
    /// # Notes
    ///
    /// - **Fourier Transform**: Converts N-dimensional time-domain signals to frequency-domain signals.
    /// - **Multidimensional**: Operates over multiple axes at once.
    ///
    /// # See Also
    ///
    /// - [`ifftn`]: Computes the N-dimensional inverse FFT of the tensor.
    /// - [`fft2`]: Computes the 2D FFT of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fftn<A: Into<Axis>>(&self, axes: A) -> Result<Self>;

    /// Computes the N-dimensional inverse Fast Fourier Transform (IFFTN) of the tensor.
    ///
    /// The `ifftn` function computes the N-dimensional inverse discrete Fourier Transform of the input tensor,
    /// converting the signal from the frequency domain back to the time domain in N dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatType<T>>>`: A tensor of real numbers representing the time-domain signal.
    ///
    /// # Notes
    ///
    /// - **Inverse Fourier Transform**: Converts N-dimensional frequency-domain signals back to the time domain.
    /// - **Multidimensional**: Operates over multiple axes at once.
    ///
    /// # See Also
    ///
    /// - [`fftn`]: Computes the N-dimensional FFT of the tensor.
    /// - [`ifft2`]: Computes the 2D inverse FFT of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifftn<A: Into<Axis>>(&self, axes: A) -> Result<Self>;
}
