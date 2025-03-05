use hpt_common::error::base::TensorError;
use hpt_types::type_promote::FloatOutBinary;

/// A trait contains window operations
pub trait WindowOps {
    /// The type of the output tensor
    type Output;
    /// The type of the meta data
    type Meta: FloatOutBinary;
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
    fn hamming_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError>;
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
    fn hann_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError>;
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
    fn blackman_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError>;
}
