use std::fmt::Debug;
use std::{borrow::Borrow, fmt::Display};
use tensor_common::error::base::TensorError;
use tensor_common::{axis::axis::Axis, layout::layout::Layout, utils::pointer::Pointer, shape::shape::Shape, strides::strides::Strides};
#[cfg(feature = "archsimd")]
use tensor_types::arch_simd as simd;
#[cfg(feature = "stdsimd")]
use tensor_types::std_simd as simd;
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut, NormalOutUnary},
};

#[cfg(target_feature = "avx2")]
type BoolVector = simd::_256bit::boolx32::boolx32;
#[cfg(target_feature = "avx512f")]
type BoolVector = simd::_512bit::boolx64::boolx64;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
type BoolVector = simd::_128bit::boolx16::boolx16;

/// A trait for getting information of a Tensor
pub trait TensorInfo<T> {
    /// Returns a pointer to the tensor's first data.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ptr(&self) -> Pointer<T>;

    /// Returns the size of the tensor based on the shape
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn size(&self) -> usize;

    /// Returns the shape of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shape(&self) -> &Shape;

    /// Returns the strides of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn strides(&self) -> &Strides;

    /// Returns the layout of the tensor. Layout contains shape and strides.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn layout(&self) -> &Layout;
    /// Returns the root tensor, if any.
    ///
    /// if the tensor is a view, it will return the root tensor. Otherwise, it will return None.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn parent(&self) -> Option<Pointer<T>>;

    /// Returns the number of dimensions of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ndim(&self) -> usize;

    /// Returns whether the tensor is contiguous in memory. View or transpose tensors are not contiguous.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn is_contiguous(&self) -> bool;

    /// Returns the data type memory size in bytes.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn elsize() -> usize {
        size_of::<T>()
    }
}

/// A trait for let the object like a tensor
pub trait TensorLike<T>: Sized {
    /// directly convert the tensor to raw slice
    ///
    /// # Note
    ///
    /// This function will return a raw slice of the tensor regardless of the shape and strides.
    ///
    /// if you do iteration on the view tensor, you may see unexpected results.
    fn as_raw(&self) -> &[T];

    /// directly convert the tensor to mutable raw slice
    ///
    /// # Note
    ///
    /// This function will return a mutable raw slice of the tensor regardless of the shape and strides.
    ///
    /// if you do iteration on the view tensor, you may see unexpected results.
    fn as_raw_mut(&mut self) -> &mut [T];

    /// Returns the tensor as a contiguous tensor.
    ///
    /// # Note
    ///
    /// This function will return a contiguous tensor. If the tensor is already contiguous, it will return a clone of the tensor.
    ///
    /// If the tensor is a view tensor, it will return a new tensor with the same data but with a contiguous layout.
    fn contiguous(&self) -> Result<Self, TensorError>;

    /// Returns the data type memory size in bytes.
    fn elsize() -> usize {
        size_of::<T>()
    }
}

/// A trait defines a set of functions to create tensors.
pub trait TensorCreator<T>
where
    Self: Sized,
{
    /// the output type of the creator
    type Output;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
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
    #[cfg_attr(feature = "track_caller", track_caller)]
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ones<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>
    where
        u8: IntoScalar<T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
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
    #[cfg_attr(feature = "track_caller", track_caller)]
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ones_like(&self) -> Result<Self::Output, TensorError>
    where
        u8: IntoScalar<T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self::Output, TensorError>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn full_like(&self, val: T) -> Result<Self::Output, TensorError>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn arange<U>(start: U, end: U) -> Result<Self::Output, TensorError>
    where
        T: Convertor + FromScalar<U> + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn arange_step(start: T, end: T, step: T) -> Result<Self::Output, TensorError>
    where
        T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError>
    where
        u8: IntoScalar<T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn linspace<U>(start: U, end: U, num: usize, include_end: bool) ->  Result<Self::Output, TensorError>
    where
        T: Convertor,
        U: Convertor + IntoScalar<T> + Copy,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self::Output, TensorError>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64> + NormalOut<T, Output = T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self::Output, TensorError>
    where
        f64: IntoScalar<T>,
        usize: IntoScalar<T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self::Output, TensorError>
    where
        u8: IntoScalar<T>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tril(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn triu(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn identity(n: usize) -> Result<Self::Output, TensorError>
    where
        u8: IntoScalar<T>;
}

/// A trait for tensor memory allocation, this trait only used when we work with generic type
pub trait TensorAlloc<Output = Self> {
    /// The tensor data type.
    type Meta;
    /// Creates a tensor with the specified shape,
    ///
    /// # Note
    ///
    /// This function doesn't initialize the tensor's elements.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn _empty<S: Into<Shape>>(shape: S) -> Result<Output, TensorError>
    where
        Self: Sized;
}

/// A trait typically for argmax and argmin functions.
pub trait IndexReduce
where
    Self: Sized,
{
    /// The output tensor type.
    type Output;

    /// Returns the indices of the maximum values along the specified axis.
    ///
    /// The `argmax` function computes the index of the maximum value along the given axis for each slice of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the index of the maximum value.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the indices of the maximum values.
    ///
    /// # See Also
    ///
    /// - [`argmin`]: Returns the indices of the minimum values along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Returns the indices of the minimum values along the specified axis.
    ///
    /// The `argmin` function computes the index of the minimum value along the given axis for each slice of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the index of the minimum value.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the indices of the minimum values.
    ///
    /// # See Also
    ///
    /// - [`argmax`]: Returns the indices of the maximum values along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;
}

/// A trait for normal tensor reduction operations.
pub trait NormalReduce<T>
where
    Self: Sized,
{
    /// The output tensor type.
    type Output;

    /// Computes the sum of the elements along the specified axis.
    ///
    /// The `sum` function computes the sum of elements along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to sum the elements.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`nansum`]: Computes the sum while ignoring NaN values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Computes the sum of the elements along the specified axis, storing the result in a pre-allocated tensor.
    ///
    /// The `sum_` function computes the sum of elements along the specified axis, and optionally initializes an output tensor to store the result.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to sum the elements.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    /// - `init_out`: Whether to initialize the output tensor.
    /// - `out`: The tensor in which to store the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements, with the result stored in the specified output tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sum_<S: Into<Axis>, O>(
        &self,
        axis: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> Result<Self::Output, TensorError>
    where
        O: Borrow<Self::Output>;

    // /// Computes the sum of the elements along the specified axis, with an initial value.
    // ///
    // /// The `sum_with_init` function computes the sum of elements along the specified axes, starting from a given initial value.
    // ///
    // /// # Parameters
    // ///
    // /// - `init_val`: The initial value to start the summation.
    // /// - `axes`: The axes along which to sum the elements.
    // /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    // ///
    // /// # Returns
    // ///
    // /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements along the specified axes.
    // #[cfg_attr(feature = "track_caller", track_caller)]
    // fn sum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output>;

    /// Computes the product of the elements along the specified axis.
    ///
    /// The `prod` function computes the product of elements along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the product.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the product of elements along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`nanprod`]: Computes the product while ignoring NaN values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    // /// Computes the product of the elements along the specified axis, with an initial value.
    // ///
    // /// The `prod_with_init` function computes the product of elements along the specified axes, starting from a given initial value.
    // ///
    // /// # Parameters
    // ///
    // /// - `init_val`: The initial value to start the product computation.
    // /// - `axes`: The axes along which to compute the product.
    // /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    // ///
    // /// # Returns
    // ///
    // /// - `anyhow::Result<Self::Output>`: A tensor containing the product of elements along the specified axes.
    // #[cfg_attr(feature = "track_caller", track_caller)]
    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output>;

    /// Computes the minimum value along the specified axis.
    ///
    /// The `min` function returns the minimum value of the elements along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the minimum value.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self>`: A tensor containing the minimum values along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`max`]: Computes the maximum value along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    // /// Computes the minimum value along the specified axis, with an initial value.
    // ///
    // /// The `min_with_init` function computes the minimum value along the specified axes, starting from a given initial value.
    // ///
    // /// # Parameters
    // ///
    // /// - `init_val`: The initial value to compare against.
    // /// - `axes`: The axes along which to compute the minimum value.
    // /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    // ///
    // /// # Returns
    // ///
    // /// - `anyhow::Result<Self>`: A tensor containing the minimum values along the specified axes.
    // #[cfg_attr(feature = "track_caller", track_caller)]
    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self>;

    /// Computes the maximum value along the specified axis.
    ///
    /// The `max` function returns the maximum value of the elements along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the maximum value.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self>`: A tensor containing the maximum values along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`min`]: Computes the minimum value along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    // /// Computes the maximum value along the specified axis, with an initial value.
    // ///
    // /// The `max_with_init` function computes the maximum value along the specified axes, starting from a given initial value.
    // ///
    // /// # Parameters
    // ///
    // /// - `init_val`: The initial value to compare against.
    // /// - `axes`: The axes along which to compute the maximum value.
    // /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    // ///
    // /// # Returns
    // ///
    // /// - `anyhow::Result<Self>`: A tensor containing the maximum values along the specified axes.
    // #[cfg_attr(feature = "track_caller", track_caller)]
    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self>;

    /// Reduces the tensor along the specified axis using the L1 norm (sum of absolute values).
    ///
    /// The `reducel1` function computes the L1 norm (sum of absolute values) along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to reduce the tensor.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor with the L1 norm computed along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Computes the sum of the squares of the elements along the specified axis.
    ///
    /// The `sum_square` function computes the sum of the squares of the elements along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to sum the squares.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of squares of elements along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;
}

/// A trait for tensor reduction operations, the output must be a boolean tensor.
pub trait EvalReduce {
    /// The boolean tensor type.
    type BoolOutput;
    /// Returns `true` if all elements along the specified axis evaluate to `true`.
    ///
    /// The `all` function checks whether all elements along the specified axis evaluate to `true`.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to check.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::BoolOutput>`: A boolean tensor indicating whether all elements evaluate to `true`.
    ///
    /// # See Also
    ///
    /// - [`any`]: Returns `true` if any element along the specified axis evaluates to `true`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn all<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::BoolOutput, TensorError>;

    /// Returns `true` if any element along the specified axis evaluates to `true`.
    ///
    /// The `any` function checks whether any element along the specified axis evaluates to `true`.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to check.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::BoolOutput>`: A boolean tensor indicating whether any element evaluates to `true`.
    ///
    /// # See Also
    ///
    /// - [`all`]: Returns `true` if all elements along the specified axis evaluate to `true`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::BoolOutput, TensorError>;
}

/// A trait for tensor reduction operations, the output must remain the same tensor type.
pub trait NormalEvalReduce<T> {
    /// the output tensor type.
    type Output;
    /// Computes the sum of the elements along the specified axis, ignoring NaN values.
    ///
    /// The `nansum` function computes the sum of elements along the specified axis, while ignoring NaN values in the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to sum the elements.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements, ignoring NaN values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn nansum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    // /// Computes the sum of the elements along the specified axis, with an initial value, ignoring NaN values.
    // ///
    // /// The `nansum_with_init` function computes the sum of elements along the specified axes, starting from a given initial value and ignoring NaN values.
    // ///
    // /// # Parameters
    // ///
    // /// - `init_val`: The initial value to start the summation.
    // /// - `axes`: The axes along which to sum the elements.
    // /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    // ///
    // /// # Returns
    // ///
    // /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements, ignoring NaN values.
    // #[cfg_attr(feature = "track_caller", track_caller)]
    // fn nansum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output>;

    /// Computes the product of the elements along the specified axis, ignoring NaN values.
    ///
    /// The `nanprod` function computes the product of elements along the specified axis, while ignoring NaN values in the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the product.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the product of elements, ignoring NaN values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    // /// Computes the product of the elements along the specified axis, with an initial value, ignoring NaN values.
    // ///
    // /// The `nanprod_with_init` function computes the product of elements along the specified axes, starting from a given initial value and ignoring NaN values.
    // ///
    // /// # Parameters
    // ///
    // /// - `init_val`: The initial value to start the product computation.
    // /// - `axes`: The axes along which to compute the product.
    // /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    // ///
    // /// # Returns
    // ///
    // /// - `anyhow::Result<Self::Output>`: A tensor containing the product of elements, ignoring NaN values.
    // #[cfg_attr(feature = "track_caller", track_caller)]
    // fn nanprod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output>;
}

/// A trait for tensor reduction operations, the output must be a floating-point tensor.
pub trait FloatReduce<T>
where
    Self: Sized,
{
    /// The output tensor type.
    type Output;

    /// Computes the mean of the elements along the specified axis.
    ///
    /// The `mean` function calculates the mean of the elements along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the mean.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the mean values along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mean<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

    /// Reduces the tensor along the specified axis using the L2 norm (Euclidean norm).
    ///
    /// The `reducel2` function computes the L2 norm (Euclidean norm) along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to reduce the tensor.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor with the L2 norm computed along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reducel2<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

    /// Reduces the tensor along the specified axis using the L3 norm.
    ///
    /// The `reducel3` function computes the L3 norm along the specified axis of the tensor.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to reduce the tensor.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor with the L3 norm computed along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reducel3<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

    /// Computes the logarithm of the sum of exponentials of the elements along the specified axis.
    ///
    /// The `logsumexp` function calculates the logarithm of the sum of exponentials of the elements along the specified axis,
    /// which is useful for numerical stability in certain operations.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the logarithm of the sum of exponentials.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the log-sum-exp values along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn logsumexp<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;
}

/// Common bounds for primitive types
pub trait CommonBounds
where
    <Self as TypeCommon>::Vec: Send + Sync + Copy,
    Self: Sync
        + Send
        + Clone
        + Copy
        + TypeCommon
        + 'static
        + Display
        + Debug
        + IntoScalar<Self>
        + Convertor
        + NormalOut<Self, Output = Self>
        + FloatOutUnary
        + NormalOut<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<Self>
        + NormalOut<
            <Self as FloatOutBinary<Self>>::Output,
            Output = <Self as FloatOutBinary<Self>>::Output,
        >
        + NormalOutUnary,
{
}
impl<T> CommonBounds for T
where
    <Self as TypeCommon>::Vec: Send + Sync + Copy,
    Self: Sync
        + Send
        + Clone
        + Copy
        + TypeCommon
        + 'static
        + Display
        + Debug
        + IntoScalar<Self>
        + Convertor
        + NormalOut<Self, Output = Self>
        + FloatOutUnary
        + NormalOut<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<Self>
        + FloatOutBinary<
            <Self as FloatOutBinary<Self>>::Output,
            Output = <Self as FloatOutBinary<Self>>::Output,
        >
        + NormalOut<
            <Self as FloatOutBinary<Self>>::Output,
            Output = <Self as FloatOutBinary<Self>>::Output,
        >
        + NormalOutUnary,
{
}
