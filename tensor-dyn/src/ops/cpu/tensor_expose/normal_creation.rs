use std::ops::{Div, Sub};

use crate::{
    backend::Cpu, tensor::Tensor, tensor_base::_Tensor,
    tensor_internal::float_out_unary::FloatUnaryType, BoolVector,
};
use anyhow::Result;
use tensor_common::shape::Shape;
use tensor_traits::{CommonBounds, TensorCreator};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{FloatOutUnary, NormalOut},
};

impl<T: CommonBounds> TensorCreator<T> for Tensor<T> {
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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::empty([2, 3]);
    /// ```
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::empty(shape)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::zeros([2, 3]);
    /// ```
    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::zeros(shape)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::ones([2, 3]);
    /// ```
    fn ones<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::ones(shape)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::empty([2, 3]);
    /// let b = a.empty_like();
    /// ```
    fn empty_like(&self) -> Result<Self> {
        Ok(_Tensor::empty_like(self)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::empty([2, 3]);
    /// let b = a.zeros_like();
    /// ```
    fn zeros_like(&self) -> Result<Self> {
        Ok(_Tensor::zeros_like(self)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::empty([2, 3]);
    /// let b = a.ones_like();
    /// ```
    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::ones_like(self)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::full(3.0, [2, 3]);
    /// ```
    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::full(val, shape)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::empty([2, 3]);
    /// let b = a.full_like(3.0);
    /// ```
    fn full_like(&self, val: T) -> Result<Self> {
        Ok(_Tensor::full_like(self, val)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::arange(0.0, 5.0).unwrap();
    /// ```
    fn arange<U>(start: U, end: U) -> Result<Self>
    where
        T: FromScalar<U>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        Ok(_Tensor::<T, Cpu>::arange(start, end)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::arange_step(0.0, 5.0, 1.0).unwrap();
    /// ```
    fn arange_step(start: T, end: T, step: T) -> Result<Self>
    where
        T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>,
    {
        Ok(_Tensor::<T, Cpu>::arange_step(start, end, step)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::eye(3, 3, 0).unwrap();
    /// ```
    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::eye(n, m, k)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::linspace(0.0, 5.0, 5, true).unwrap();
    /// ```
    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::linspace(start, end, num, include_end)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::logspace(0.0, 5.0, 5, true, 10.0).unwrap();
    /// ```
    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64> + NormalOut<T, Output = T>,
    {
        Ok(_Tensor::<T, Cpu>::logspace(start, end, num, include_end, base)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::geomspace(1.0, 1000.0, 4, true).unwrap();
    /// ```
    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        T: PartialOrd + FloatOutUnary + FromScalar<FloatUnaryType<T>> + std::ops::Neg<Output = T>,
        FloatUnaryType<T>: Sub<Output = FloatUnaryType<T>>
            + FromScalar<usize>
            + FromScalar<f64>
            + Div<Output = FloatUnaryType<T>>
            + CommonBounds,
    {
        Ok(_Tensor::<T, Cpu>::geomspace(start, end, n, include_end)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::tri(3, 3, 0, true).unwrap();
    /// ```
    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::tri(n, m, k, low_triangle)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::eye(3, 3, 0).unwrap();
    /// let b = a.tril(1).unwrap();
    /// ```
    fn tril(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(_Tensor::tril(self, k)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::eye(3, 3, 0).unwrap();
    /// let b = a.triu(1).unwrap();
    /// ```
    fn triu(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        Ok(_Tensor::triu(self, k)?.into())
    }

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
    /// # Example
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::TensorCreator;
    /// let a = Tensor::<f64>::identity(3).unwrap();
    /// ```
    fn identity(n: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::identity(n)?.into())
    }
}
