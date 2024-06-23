use anyhow::Result;
use tensor_common::axis::Axis;

pub trait FFTOps where Self: Sized {
    /// Performs a Fast Fourier Transform (FFT) along a specified axis.
    ///
    /// This method computes the FFT along a single axis of the tensor. It delegates to the `fftn` method,
    /// which handles the actual computation. This is a convenience method for computing FFT along one dimension.
    ///
    /// # Arguments
    /// - `axis`: The axis along which to compute the FFT. It is an `isize`, allowing for both positive
    ///   and negative indexing.
    ///
    /// # Returns
    /// `anyhow::Result<Self>`: The result of the FFT computation along the specified axis.
    /// It returns an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex32;
    /// use tensor_core::Tensor;
    /// use tensor_trait::FFTOps;
    /// let tensor = Tensor::<Complex32>::new([1.0, 2.0, 3.0]);
    /// let result = tensor.fft(0); // Compute FFT along the first axis
    /// ```
    fn fft(&self, axis: i64) -> Result<Self>;

    /// Performs an Inverse Fast Fourier Transform (IFFT) along a specified axis.
    ///
    /// This method computes the IFFT along a single axis of the tensor. Similar to `fft`, it calls
    /// the `ifftn` method for the actual computation. Useful for computing IFFT in one-dimensional cases.
    ///
    /// # Arguments
    /// - `axis`: The axis along which to compute the IFFT. It is an `isize`, which allows for both positive
    ///   and negative indexing.
    ///
    /// # Returns
    /// `anyhow::Result<Self>`: The result of the IFFT computation along the specified axis.
    /// It returns an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex32;
    /// use tensor_core::Tensor;
    /// use tensor_trait::FFTOps;
    /// let tensor = Tensor::<Complex32>::new([1.0, 2.0, 3.0]);
    /// let result = tensor.ifft(0); // Compute IFFT along the first axis
    /// ```
    fn ifft(&self, axis: i64) -> Result<Self>;

    /// Performs a Fast Fourier Transform (FFT) along two specified axes.
    ///
    /// This method computes the FFT along two axes of the tensor. It delegates to the `fftn` method,
    /// which handles the actual computation. This is a convenience method for computing FFT along one dimension.
    ///
    /// # Arguments
    /// - `axis`: The axis along which to compute the FFT. It is an `isize`, allowing for both positive
    ///   and negative indexing.
    ///
    /// # Returns
    /// `anyhow::Result<Self>`: The result of the FFT computation along the specified axis.
    /// It returns an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex32;
    /// use tensor_core::Tensor;
    /// use tensor_trait::FFTOps;
    /// let tensor = Tensor::<Complex32>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let result = tensor.fft2([0, 1]); // Compute FFT along all the axis
    /// ```
    fn fft2(&self, axis1: i64, axis2: i64) -> Result<Self>;

    /// Performs an Inverse Fast Fourier Transform (FFT) along two specified axes.
    ///
    /// This method computes the inverse FFT along two axes of the tensor. It delegates to the `fftn` method,
    /// which handles the actual computation. This is a convenience method for computing FFT along one dimension.
    ///
    /// # Arguments
    /// - `axis`: The axis along which to compute the FFT. It is an `isize`, allowing for both positive
    ///   and negative indexing.
    ///
    /// # Returns
    /// `anyhow::Result<Self>`: The result of the FFT computation along the specified axis.
    /// It returns an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex32;
    /// use tensor_core::Tensor;
    /// use tensor_trait::FFTOps;
    /// let tensor = Tensor::<Complex32>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let result = tensor.ifft2([0, 1]); // Compute inverse FFT along all the axis
    /// ```
    fn ifft2(&self, axis1: i64, axis2: i64) -> Result<Self>;

    /// Performs an N-dimensional Fast Fourier Transform (FFT).
    ///
    /// This method computes the FFT along multiple axes of the tensor. It handles complex FFT logic,
    /// including transposing and parallel processing, to efficiently compute the FFT over specified axes.
    ///
    /// # Arguments
    /// - `axes`: The axes along which to compute the FFT. This is generic and can be converted into `Axis`.
    ///
    /// # Returns
    /// `anyhow::Result<Self>`: The result of the FFT computation along the specified axes.
    /// Returns an error if the operation fails.
    ///
    /// # Errors
    /// This method returns an error if any issue occurs during the FFT computation, such as invalid axes.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex32;
    /// use tensor_core::Tensor;
    /// use tensor_trait::FFTOps;
    /// let tensor = Tensor::<Complex32>::new([1.0, 2.0, 3.0]);
    /// let fft_result = tensor.fftn(vec![0, 1]); // Compute FFT along axes 0 and 1
    /// ```
    fn fftn<A: Into<Axis>>(&self, axes: A) -> Result<Self>;

    /// Performs an N-dimensional Inverse Fast Fourier Transform (IFFT).
    ///
    /// This method computes the IFFT along multiple axes of the tensor. It manages complex IFFT logic,
    /// including transposing and parallel processing, to efficiently compute the IFFT over specified axes.
    ///
    /// # Type Parameters
    /// - `A`: A type that can be converted into the `Axis` type. It represents the axes along which IFFT is computed.
    ///
    /// # Arguments
    /// - `axes`: The axes along which to compute the IFFT. This is generic and can be converted into `Axis`.
    ///
    /// # Returns
    /// `anyhow::Result<Self>`: The result of the IFFT computation along the specified axes.
    /// Returns an error if the operation fails.
    ///
    /// # Errors
    /// This method returns an error if any issue occurs during the IFFT computation, such as invalid axes.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex32;
    /// use tensor_core::Tensor;
    /// use tensor_trait::FFTOps;
    /// let tensor = Tensor::<Complex32>::new([1.0, 2.0, 3.0]);
    /// let ifft_result = tensor.ifftn(vec![0, 1]); // Compute IFFT along axes 0 and 1
    /// ```
    fn ifftn<A: Into<Axis>>(&self, axes: A) -> Result<Self>;
}
