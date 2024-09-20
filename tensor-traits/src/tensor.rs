use std::{
    fmt::Display,
    ops::{Div, Sub},
};

use tensor_common::{axis::Axis, layout::Layout, pointer::Pointer, shape::Shape, strides::Strides};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{FloatOutUnary, NormalOut},
};

#[cfg(target_feature = "avx2")]
type BoolVector = tensor_types::_256bit::boolx32::boolx32;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
type BoolVector = tensor_types::_512bit::boolx64::boolx64;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
type BoolVector = tensor_types::_128bit::boolx16::boolx16;

pub trait TensorInfo<T> {
    /// Returns a pointer to the tensor's first data.
    fn ptr(&self) -> Pointer<T>;

    /// Returns the size of the tensor based on the shape
    fn size(&self) -> usize;

    /// Returns the shape of the tensor.
    fn shape(&self) -> &Shape;

    /// Returns the strides of the tensor.
    fn strides(&self) -> &Strides;

    /// Returns the layout of the tensor. Layout contains shape and strides.
    fn layout(&self) -> &Layout;
    /// Returns the root tensor, if any.
    ///
    /// if the tensor is a view, it will return the root tensor. Otherwise, it will return None.
    fn parent(&self) -> Option<Pointer<T>>;

    /// Returns the number of dimensions of the tensor.
    fn ndim(&self) -> usize;

    /// Returns whether the tensor is contiguous in memory. View or transpose tensors are not contiguous.
    fn is_contiguous(&self) -> bool;

    /// Returns the data type memory size in bytes.
    fn elsize() -> usize {
        size_of::<T>()
    }
}

pub trait TensorLike<T> {
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

    /// Returns the data type memory size in bytes.
    fn elsize() -> usize {
        size_of::<T>()
    }
}

pub trait TensorCreator<T, Output = Self>
where
    Self: Sized,
{
    type StridedIter;
    type Mask;
    type Basic;

    /// Creates an uninitialized tensor with the specified shape.
    ///
    /// The `empty` function creates a tensor with the specified shape without initializing its elements.
    /// The values in the tensor will be undefined.
    ///
    /// # Parameters
    ///
    /// - `shape`: The shape of the tensor to create. It can be a fixed-size array or a vector representing the dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified shape and undefined values.
    ///
    /// # See Also
    ///
    /// - [`zeros`]: Creates a tensor filled with zeros.
    /// - [`ones`]: Creates a tensor filled with ones.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Output>;

    /// Creates a tensor filled with zeros, with the specified shape.
    ///
    /// The `zeros` function creates a tensor with the specified shape, and all elements are initialized to zero.
    ///
    /// # Parameters
    ///
    /// - `shape`: The shape of the tensor to create. It can be a fixed-size array or a vector representing the dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor filled with zeros.
    ///
    /// # See Also
    ///
    /// - [`empty`]: Creates an uninitialized tensor with the specified shape.
    /// - [`ones`]: Creates a tensor filled with ones.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zeros<S: Into<Shape>>(shape: S) -> anyhow::Result<Output>;

    /// Creates a tensor filled with ones, with the specified shape.
    ///
    /// The `ones` function creates a tensor with the specified shape, and all elements are initialized to one.
    ///
    /// # Parameters
    ///
    /// - `shape`: The shape of the tensor to create. It can be a fixed-size array or a vector representing the dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor filled with ones.
    ///
    /// # See Also
    ///
    /// - [`zeros`]: Creates a tensor filled with zeros.
    /// - [`full`]: Creates a tensor filled with a specific value.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ones<S: Into<Shape>>(shape: S) -> anyhow::Result<Output>
    where
        u8: IntoScalar<T>;

    /// Creates an uninitialized tensor with the same shape as the current tensor.
    ///
    /// The `empty_like` function creates a tensor with the same shape as the input tensor, but without initializing its elements.
    /// The values in the tensor will be undefined.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the same shape as the input tensor and undefined values.
    ///
    /// # See Also
    ///
    /// - [`zeros_like`]: Creates a tensor with the same shape and filled with zeros.
    /// - [`ones_like`]: Creates a tensor with the same shape and filled with ones.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn empty_like(&self) -> anyhow::Result<Output>;

    /// Creates a tensor filled with zeros, with the same shape as the current tensor.
    ///
    /// The `zeros_like` function creates a tensor with the same shape as the input tensor, and all elements are initialized to zero.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor filled with zeros and with the same shape as the input tensor.
    ///
    /// # See Also
    ///
    /// - [`empty_like`]: Creates an uninitialized tensor with the same shape.
    /// - [`ones_like`]: Creates a tensor filled with ones and with the same shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zeros_like(&self) -> anyhow::Result<Output>;

    /// Creates a tensor filled with ones, with the same shape as the current tensor.
    ///
    /// The `ones_like` function creates a tensor with the same shape as the input tensor, and all elements are initialized to one.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor filled with ones and with the same shape as the input tensor.
    ///
    /// # See Also
    ///
    /// - [`zeros_like`]: Creates a tensor filled with zeros and with the same shape.
    /// - [`full_like`]: Creates a tensor filled with a specific value and with the same shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ones_like(&self) -> anyhow::Result<Output>
    where
        u8: IntoScalar<T>;

    /// Creates a tensor filled with a specified value, with the specified shape.
    ///
    /// The `full` function creates a tensor with the specified shape, and all elements are initialized to the specified value.
    ///
    /// # Parameters
    ///
    /// - `val`: The value to fill the tensor with.
    /// - `shape`: The shape of the tensor to create. It can be a fixed-size array or a vector representing the dimensions.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor filled with the specified value.
    ///
    /// # See Also
    ///
    /// - [`zeros`]: Creates a tensor filled with zeros.
    /// - [`ones`]: Creates a tensor filled with ones.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn full<S: Into<Shape>>(val: T, shape: S) -> anyhow::Result<Output>;

    /// Creates a tensor filled with a specified value, with the same shape as the current tensor.
    ///
    /// The `full_like` function creates a tensor with the same shape as the input tensor, and all elements are initialized to the specified value.
    ///
    /// # Parameters
    ///
    /// - `val`: The value to fill the tensor with.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor filled with the specified value and with the same shape as the input tensor.
    ///
    /// # See Also
    ///
    /// - [`zeros_like`]: Creates a tensor filled with zeros and with the same shape.
    /// - [`ones_like`]: Creates a tensor filled with ones and with the same shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn full_like(&self, val: T) -> anyhow::Result<Output>;

    /// Generates a tensor with evenly spaced values between `start` and `end`.
    ///
    /// The `arange` function creates a tensor of values in the range `[start, end)`, with a step size of 1.
    ///
    /// # Parameters
    ///
    /// - `start`: The starting value of the range.
    /// - `end`: The end value of the range (exclusive).
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A tensor with evenly spaced values between `start` and `end`.
    ///
    /// # See Also
    ///
    /// - [`arange_step`]: Generates a tensor with evenly spaced values between `start` and `end`, with a custom step size.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn arange<U>(start: U, end: U) -> anyhow::Result<Output>
    where
        T: Convertor + FromScalar<U> + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy;

    /// Generates a tensor with evenly spaced values between `start` and `end`, with a specified step size.
    ///
    /// The `arange_step` function creates a tensor of values in the range `[start, end)`, with the specified `step` size.
    ///
    /// # Parameters
    ///
    /// - `start`: The starting value of the range.
    /// - `end`: The end value of the range (exclusive).
    /// - `step`: The step size between consecutive values.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A tensor with evenly spaced values between `start` and `end` with the specified step size.
    ///
    /// # See Also
    ///
    /// - [`arange`]: Generates a tensor with evenly spaced values with a default step size of 1.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn arange_step(start: T, end: T, step: T) -> anyhow::Result<Output>
    where
        T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>;

    /// Generates a 2D identity matrix with ones on the diagonal and zeros elsewhere.
    ///
    /// The `eye` function creates a matrix with `n` rows and `m` columns, with ones on the `k`-th diagonal and zeros elsewhere.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of rows.
    /// - `m`: The number of columns.
    /// - `k`: The index of the diagonal where ones should appear (0 refers to the main diagonal).
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A 2D identity matrix.
    ///
    /// # See Also
    ///
    /// - [`identity`]: Generates a square identity matrix of size `n x n`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn eye(n: usize, m: usize, k: usize) -> anyhow::Result<Output>
    where
        u8: IntoScalar<T>;

    /// Generates a tensor with `num` evenly spaced values between `start` and `end`.
    ///
    /// The `linspace` function creates a tensor of `num` values, evenly spaced between `start` and `end`. Optionally, the end value can be included.
    ///
    /// # Parameters
    ///
    /// - `start`: The starting value of the range.
    /// - `end`: The end value of the range.
    /// - `num`: The number of values to generate.
    /// - `include_end`: Whether to include the `end` value in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A tensor with `num` evenly spaced values between `start` and `end`.
    ///
    /// # See Also
    ///
    /// - [`logspace`]: Generates a tensor with logarithmically spaced values.
    /// - [`geomspace`]: Generates a tensor with geometrically spaced values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn linspace(start: T, end: T, num: usize, include_end: bool) -> anyhow::Result<Output>
    where
        T: Convertor + num::Float + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>;

    /// Generates a tensor with `num` logarithmically spaced values between `start` and `end`.
    ///
    /// The `logspace` function creates a tensor of `num` values, logarithmically spaced between `start` and `end`. The spacing is based on the specified `base`.
    ///
    /// # Parameters
    ///
    /// - `start`: The starting value (in logarithmic scale).
    /// - `end`: The end value (in logarithmic scale).
    /// - `num`: The number of values to generate.
    /// - `include_end`: Whether to include the `end` value in the result.
    /// - `base`: The base of the logarithm used for spacing.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A tensor with `num` logarithmically spaced values.
    ///
    /// # See Also
    ///
    /// - [`linspace`]: Generates a tensor with evenly spaced values.
    /// - [`geomspace`]: Generates a tensor with geometrically spaced values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> anyhow::Result<Output>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64> + NormalOut<T, Output = T>;

    /// Generates a tensor with `n` geometrically spaced values between `start` and `end`.
    ///
    /// The `geomspace` function creates a tensor of `n` values, geometrically spaced between `start` and `end`.
    ///
    /// # Parameters
    ///
    /// - `start`: The starting value of the range.
    /// - `end`: The end value of the range.
    /// - `n`: The number of values to generate.
    /// - `include_end`: Whether to include the `end` value in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A tensor with `n` geometrically spaced values between `start` and `end`.
    ///
    /// # See Also
    ///
    /// - [`linspace`]: Generates a tensor with evenly spaced values.
    /// - [`logspace`]: Generates a tensor with logarithmically spaced values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> anyhow::Result<Output>
    where
        T: PartialOrd
            + FloatOutUnary
            + NormalOut<T, Output = T>
            + FromScalar<<T as FloatOutUnary>::Output>
            + std::ops::Neg<Output = T>,
        <T as FloatOutUnary>::Output: Sub<Output = <T as FloatOutUnary>::Output>
            + FromScalar<usize>
            + FromScalar<f64>
            + Div<Output = <T as FloatOutUnary>::Output>
            + NormalOut<Output = <T as FloatOutUnary>::Output>
            + CommonBounds,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec: Send + Sync;

    /// Generates a lower or upper triangular matrix.
    ///
    /// The `tri` function creates a matrix with `n` rows and `m` columns, where elements below or above the `k`-th diagonal are set to zero, depending on the `low_triangle` flag.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of rows.
    /// - `m`: The number of columns.
    /// - `k`: The index of the diagonal where the triangle begins (0 refers to the main diagonal).
    /// - `low_triangle`: A boolean indicating whether to return the lower or upper triangular matrix.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A lower or upper triangular matrix.
    ///
    /// # See Also
    ///
    /// - [`tril`]: Returns the lower triangular part of a matrix.
    /// - [`triu`]: Returns the upper triangular part of
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> anyhow::Result<Output>
    where
        u8: IntoScalar<T>;

    /// Returns the lower triangular part of the tensor, setting elements above the `k`-th diagonal to zero.
    ///
    /// The `tril` function extracts the lower triangular part of the tensor, setting elements above the `k`-th diagonal to zero.
    ///
    /// # Parameters
    ///
    /// - `k`: The index of the diagonal (0 refers to the main diagonal).
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self>`: A tensor with the upper triangular part zeroed out.
    ///
    /// # See Also
    ///
    /// - [`tri`]: Generates a lower or upper triangular matrix.
    /// - [`triu`]: Returns the upper triangular part of a tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tril(&self, k: i64) -> anyhow::Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>;

    /// Returns the upper triangular part of the tensor, setting elements below the `k`-th diagonal to zero.
    ///
    /// The `triu` function extracts the upper triangular part of the tensor, setting elements below the `k`-th diagonal to zero.
    ///
    /// # Parameters
    ///
    /// - `k`: The index of the diagonal (0 refers to the main diagonal).
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self>`: A tensor with the lower triangular part zeroed out.
    ///
    /// # See Also
    ///
    /// - [`tri`]: Generates a lower or upper triangular matrix.
    /// - [`tril`]: Returns the lower triangular part of a tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn triu(&self, k: i64) -> anyhow::Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>;

    /// Generates a square identity matrix of size `n x n`.
    ///
    /// The `identity` function creates a matrix with `n` rows and `n` columns, with ones on the main diagonal and zeros elsewhere.
    ///
    /// # Parameters
    ///
    /// - `n`: The size of the matrix (number of rows and columns).
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A square identity matrix.
    ///
    /// # See Also
    ///
    /// - [`eye`]: Generates a matrix with ones on a specified diagonal and zeros elsewhere.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn identity(n: usize) -> anyhow::Result<Output>
    where
        u8: IntoScalar<T>;
}

pub trait TensorAlloc<Output = Self> {
    type Meta;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Output>
    where
        Self: Sized;
}

pub trait IndexReduce
where
    Self: Sized,
{
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
    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

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
    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;
}

pub trait NormalReduce<T>
where
    Self: Sized,
{
    type Output;
    type BoolOutput;

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
    fn sum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

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
    fn sum_<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
        init_out: bool,
        out: Self::Output,
    ) -> anyhow::Result<Self::Output>;

    /// Computes the sum of the elements along the specified axis, with an initial value.
    ///
    /// The `sum_with_init` function computes the sum of elements along the specified axes, starting from a given initial value.
    ///
    /// # Parameters
    ///
    /// - `init_val`: The initial value to start the summation.
    /// - `axes`: The axes along which to sum the elements.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements along the specified axes.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output>;

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
    fn nansum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

    /// Computes the sum of the elements along the specified axis, with an initial value, ignoring NaN values.
    ///
    /// The `nansum_with_init` function computes the sum of elements along the specified axes, starting from a given initial value and ignoring NaN values.
    ///
    /// # Parameters
    ///
    /// - `init_val`: The initial value to start the summation.
    /// - `axes`: The axes along which to sum the elements.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the sum of elements, ignoring NaN values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn nansum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output>;

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
    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

    /// Computes the product of the elements along the specified axis, with an initial value.
    ///
    /// The `prod_with_init` function computes the product of elements along the specified axes, starting from a given initial value.
    ///
    /// # Parameters
    ///
    /// - `init_val`: The initial value to start the product computation.
    /// - `axes`: The axes along which to compute the product.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the product of elements along the specified axes.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output>;

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
    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

    /// Computes the product of the elements along the specified axis, with an initial value, ignoring NaN values.
    ///
    /// The `nanprod_with_init` function computes the product of elements along the specified axes, starting from a given initial value and ignoring NaN values.
    ///
    /// # Parameters
    ///
    /// - `init_val`: The initial value to start the product computation.
    /// - `axes`: The axes along which to compute the product.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self::Output>`: A tensor containing the product of elements, ignoring NaN values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn nanprod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output>;

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
    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self>;

    /// Computes the minimum value along the specified axis, with an initial value.
    ///
    /// The `min_with_init` function computes the minimum value along the specified axes, starting from a given initial value.
    ///
    /// # Parameters
    ///
    /// - `init_val`: The initial value to compare against.
    /// - `axes`: The axes along which to compute the minimum value.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self>`: A tensor containing the minimum values along the specified axes.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self>;

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
    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self>;

    /// Computes the maximum value along the specified axis, with an initial value.
    ///
    /// The `max_with_init` function computes the maximum value along the specified axes, starting from a given initial value.
    ///
    /// # Parameters
    ///
    /// - `init_val`: The initial value to compare against.
    /// - `axes`: The axes along which to compute the maximum value.
    /// - `keep_dims`: Whether to retain the reduced dimensions in the result.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Self>`: A tensor containing the maximum values along the specified axes.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self>;

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
    fn all<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput>;

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
    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput>;

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
    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;

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
    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output>;
}

pub trait FloatReduce<T>
where
    Self: Sized,
{
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

pub trait CommonBounds:
    Sync + Send + Clone + Copy + TypeCommon + 'static + Display + IntoScalar<Self> + Convertor
where
    <Self as TypeCommon>::Vec: Send + Sync + Copy,
{
}
impl<
        T: Sync + Send + Clone + Copy + TypeCommon + 'static + Display + IntoScalar<Self> + Convertor,
    > CommonBounds for T
where
    <Self as TypeCommon>::Vec: Send + Sync + Copy,
{
}
