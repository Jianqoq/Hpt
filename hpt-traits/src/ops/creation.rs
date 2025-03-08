use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_types::arch_simd as simd;
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::NormalOut};
#[cfg(target_feature = "avx2")]
type BoolVector = simd::_256bit::boolx32::boolx32;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
type BoolVector = simd::_128bit::boolx16::boolx16;

/// A trait defines a set of functions to create tensors.
pub trait TensorCreator
where
    Self: Sized,
{
    /// the output type of the tensor
    type Output;
    /// the meta type of the tensor
    type Meta;
    /// Creates a tensor with uninitialized elements of the specified shape.
    ///
    /// This function allocates memory for a tensor of the given shape, but the values are uninitialized, meaning they may contain random data.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor. The type `S` must implement `Into<Shape>`.
    ///
    /// # Returns
    ///
    /// * A tensor with the specified shape, but with uninitialized data.
    ///
    /// # Panics
    ///
    /// * This function may panic if the requested shape is invalid or too large for available memory.
    #[track_caller]
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>;

    /// Creates a tensor filled with zeros of the specified shape.
    ///
    /// This function returns a tensor where every element is initialized to `0`, with the shape defined by the input.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor. The type `S` must implement `Into<Shape>`.
    ///
    /// # Returns
    ///
    /// * A tensor filled with zeros, with the specified shape.
    ///
    /// # Panics
    ///
    /// * This function may panic if the requested shape is invalid or too large for available memory.
    #[track_caller]
    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>;

    /// Creates a tensor filled with ones of the specified shape.
    ///
    /// This function returns a tensor where every element is initialized to `1`, with the shape defined by the input.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor. The type `S` must implement `Into<Shape>`.
    ///
    /// # Returns
    ///
    /// * A tensor filled with ones, with the specified shape.
    ///
    /// # Panics
    ///
    /// * This function may panic if the requested shape is invalid or too large for available memory.
    #[track_caller]
    fn ones<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;

    /// Creates a tensor with uninitialized elements, having the same shape as the input tensor.
    ///
    /// This function returns a tensor with the same shape as the calling tensor, but with uninitialized data.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A tensor with the same shape as the input, but with uninitialized data.
    ///
    /// # Panics
    ///
    /// * This function may panic if the shape is too large for available memory.
    #[track_caller]
    fn empty_like(&self) -> Result<Self::Output, TensorError>;

    /// Creates a tensor filled with zeros, having the same shape as the input tensor.
    ///
    /// This function returns a tensor with the same shape as the calling tensor, with all elements initialized to `0`.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A tensor with the same shape as the input, filled with zeros.
    ///
    /// # Panics
    ///
    /// * This function may panic if the shape is too large for available memory.
    #[track_caller]
    fn zeros_like(&self) -> Result<Self::Output, TensorError>;

    /// Creates a tensor filled with ones, having the same shape as the input tensor.
    ///
    /// This function returns a tensor with the same shape as the calling tensor, with all elements initialized to `1`.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A tensor with the same shape as the input, filled with ones.
    ///
    /// # Panics
    ///
    /// * This function may panic if the shape is too large for available memory.
    #[track_caller]
    fn ones_like(&self) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;

    /// Creates a tensor filled with a specified value, with the specified shape.
    ///
    /// This function returns a tensor where every element is set to `val`, with the shape defined by the input.
    ///
    /// # Arguments
    ///
    /// * `val` - The value to fill the tensor with.
    /// * `shape` - The desired shape of the tensor. The type `S` must implement `Into<Shape>`.
    ///
    /// # Returns
    ///
    /// * A tensor filled with `val`, with the specified shape.
    ///
    /// # Panics
    ///
    /// * This function may panic if the requested shape is invalid or too large for available memory.
    #[track_caller]
    fn full<S: Into<Shape>>(val: Self::Meta, shape: S) -> Result<Self::Output, TensorError>;

    /// Creates a tensor filled with a specified value, having the same shape as the input tensor.
    ///
    /// This function returns a tensor where every element is set to `val`, with the same shape as the calling tensor.
    ///
    /// # Arguments
    ///
    /// * `val` - The value to fill the tensor with.
    ///
    /// # Returns
    ///
    /// * A tensor with the same shape as the input, filled with `val`.
    ///
    /// # Panics
    ///
    /// * This function may panic if the shape is too large for available memory.
    #[track_caller]
    fn full_like(&self, val: Self::Meta) -> Result<Self::Output, TensorError>;

    /// Creates a tensor with values within a specified range.
    ///
    /// This function generates a 1D tensor with values ranging from `start` (inclusive) to `end` (exclusive).
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the range.
    /// * `end` - The end of the range.
    ///
    /// # Returns
    ///
    /// * A 1D tensor with values ranging from `start` to `end`.
    ///
    /// # Panics
    ///
    /// * This function will panic if `start` is greater than or equal to `end`, or if the range is too large for available memory.
    #[track_caller]
    fn arange<U>(start: U, end: U) -> Result<Self::Output, TensorError>
    where
        usize: Cast<Self::Meta>,
        U: Cast<i64> + Cast<Self::Meta> + Copy;

    /// Creates a tensor with values within a specified range with a given step size.
    ///
    /// This function generates a 1D tensor with values ranging from `start` (inclusive) to `end` (exclusive),
    /// incremented by `step`.
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the range.
    /// * `end` - The end of the range (exclusive).
    /// * `step` - The step size between consecutive values.
    ///
    /// # Returns
    ///
    /// * A 1D tensor with values from `start` to `end`, incremented by `step`.
    ///
    /// # Panics
    ///
    /// * This function will panic if `step` is zero or if the range and step values are incompatible.
    #[track_caller]
    fn arange_step(
        start: Self::Meta,
        end: Self::Meta,
        step: Self::Meta,
    ) -> Result<Self::Output, TensorError>
    where
        Self::Meta: Cast<f64> + Cast<f64>,
        f64: Cast<Self::Meta>,
        usize: Cast<Self::Meta>;

    /// Creates a 2D identity matrix with ones on a diagonal and zeros elsewhere.
    ///
    /// This function generates a matrix of size `n` by `m`, with ones on the `k`th diagonal (can be offset) and zeros elsewhere.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of rows in the matrix.
    /// * `m` - The number of columns in the matrix.
    /// * `k` - The diagonal offset (0 for main diagonal, positive for upper diagonals, negative for lower diagonals).
    ///
    /// # Returns
    ///
    /// * A 2D identity matrix with ones on the specified diagonal.
    ///
    /// # Panics
    ///
    /// * This function will panic if `n` or `m` is zero, or if memory constraints are exceeded.
    #[track_caller]
    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError>;

    /// Creates a tensor with evenly spaced values between `start` and `end`.
    ///
    /// This function generates a 1D tensor of `num` values, linearly spaced between `start` and `end`.
    /// If `include_end` is `true`, the `end` value will be included as the last element.
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the range.
    /// * `end` - The end of the range.
    /// * `num` - The number of evenly spaced values to generate.
    /// * `include_end` - Whether to include the `end` value in the generated tensor.
    ///
    /// # Returns
    ///
    /// * A 1D tensor with `num` linearly spaced values between `start` and `end`.
    ///
    /// # Panics
    ///
    /// * This function will panic if `num` is zero or if `num` is too large for available memory.
    #[track_caller]
    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        U: Cast<f64> + Cast<Self::Meta> + Copy,
        usize: Cast<Self::Meta>,
        f64: Cast<Self::Meta>;

    /// Creates a tensor with logarithmically spaced values between `start` and `end`.
    ///
    /// This function generates a 1D tensor of `num` values spaced evenly on a log scale between `start` and `end`.
    /// The spacing is based on the logarithm to the given `base`. If `include_end` is `true`, the `end` value will be included.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting exponent (base `base`).
    /// * `end` - The ending exponent (base `base`).
    /// * `num` - The number of logarithmically spaced values to generate.
    /// * `include_end` - Whether to include the `end` value in the generated tensor.
    /// * `base` - The base of the logarithm.
    ///
    /// # Returns
    ///
    /// * A 1D tensor with `num` logarithmically spaced values between `start` and `end`.
    ///
    /// # Panics
    ///
    /// * This function will panic if `num` is zero or if `base` is less than or equal to zero.
    #[track_caller]
    fn logspace(
        start: Self::Meta,
        end: Self::Meta,
        num: usize,
        include_end: bool,
        base: Self::Meta,
    ) -> Result<Self::Output, TensorError>
    where
        Self::Meta: Cast<f64> + num::Float,
        usize: Cast<Self::Meta>,
        f64: Cast<Self::Meta>;

    /// Creates a tensor with geometrically spaced values between `start` and `end`.
    ///
    /// This function generates a 1D tensor of `n` values spaced evenly on a geometric scale between `start` and `end`.
    /// If `include_end` is `true`, the `end` value will be included.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value (must be positive).
    /// * `end` - The ending value (must be positive).
    /// * `n` - The number of geometrically spaced values to generate.
    /// * `include_end` - Whether to include the `end` value in the generated tensor.
    ///
    /// # Returns
    ///
    /// * A 1D tensor with `n` geometrically spaced values between `start` and `end`.
    ///
    /// # Panics
    ///
    /// * This function will panic if `n` is zero, if `start` or `end` is negative, or if the values result in undefined behavior.
    #[track_caller]
    fn geomspace(
        start: Self::Meta,
        end: Self::Meta,
        n: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        f64: Cast<Self::Meta>,
        usize: Cast<Self::Meta>,
        Self::Meta: Cast<f64>;

    /// Creates a 2D triangular matrix of size `n` by `m`, with ones below or on the `k`th diagonal and zeros elsewhere.
    ///
    /// This function generates a matrix with a triangular structure, filled with ones and zeros, based on the diagonal offset and the `low_triangle` flag.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of rows in the matrix.
    /// * `m` - The number of columns in the matrix.
    /// * `k` - The diagonal offset (0 for main diagonal, positive for upper diagonals, negative for lower diagonals).
    /// * `low_triangle` - If `true`, the matrix will be lower triangular; otherwise, upper triangular.
    ///
    /// # Returns
    ///
    /// * A 2D triangular matrix of ones and zeros.
    ///
    /// # Panics
    ///
    /// * This function will panic if `n` or `m` is zero.
    #[track_caller]
    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;

    /// Returns the lower triangular part of the matrix, with all elements above the `k`th diagonal set to zero.
    ///
    /// This function generates a tensor where the elements above the specified diagonal are set to zero.
    ///
    /// # Arguments
    ///
    /// * `k` - The diagonal offset (0 for main diagonal, positive for upper diagonals, negative for lower diagonals).
    ///
    /// # Returns
    ///
    /// * A tensor with its upper triangular part set to zero.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[track_caller]
    fn tril(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        Self::Meta: NormalOut<bool, Output = Self::Meta> + Cast<Self::Meta> + TypeCommon,
        <Self::Meta as TypeCommon>::Vec:
            NormalOut<BoolVector, Output = <Self::Meta as TypeCommon>::Vec>;

    /// Returns the upper triangular part of the matrix, with all elements below the `k`th diagonal set to zero.
    ///
    /// This function generates a tensor where the elements below the specified diagonal are set to zero.
    ///
    /// # Arguments
    ///
    /// * `k` - The diagonal offset (0 for main diagonal, positive for upper diagonals, negative for lower diagonals).
    ///
    /// # Returns
    ///
    /// * A tensor with its lower triangular part set to zero.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[track_caller]
    fn triu(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        Self::Meta: NormalOut<bool, Output = Self::Meta> + Cast<Self::Meta> + TypeCommon,
        <Self::Meta as TypeCommon>::Vec:
            NormalOut<BoolVector, Output = <Self::Meta as TypeCommon>::Vec>;

    /// Creates a 2D identity matrix of size `n` by `n`.
    ///
    /// This function generates a square matrix with ones on the main diagonal and zeros elsewhere.
    ///
    /// # Arguments
    ///
    /// * `n` - The size of the matrix (both rows and columns).
    ///
    /// # Returns
    ///
    /// * A 2D identity matrix of size `n`.
    ///
    /// # Panics
    ///
    /// * This function will panic if `n` is zero.
    #[track_caller]
    fn identity(n: usize) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;
}
