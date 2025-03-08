use std::borrow::BorrowMut;

use hpt_common::{axis::axis::Axis, error::base::TensorError};

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
    #[track_caller]
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
    #[track_caller]
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
    #[track_caller]
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
    #[track_caller]
    fn sum_<S: Into<Axis>, O>(
        &self,
        axis: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> Result<Self::Output, TensorError>
    where
        O: BorrowMut<Self::Output>;

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
    // #[track_caller]
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
    #[track_caller]
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
    // #[track_caller]
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
    #[track_caller]
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
    // #[track_caller]
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
    #[track_caller]
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
    // #[track_caller]
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
    #[track_caller]
    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;

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
    #[track_caller]
    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;
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
    #[track_caller]
    fn all<S: Into<Axis>>(&self, axis: S, keep_dims: bool)
        -> Result<Self::BoolOutput, TensorError>;

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
    #[track_caller]
    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool)
        -> Result<Self::BoolOutput, TensorError>;
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
    #[track_caller]
    fn nansum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

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
    #[track_caller]
    fn nansum_<S: Into<Axis>, O>(
        &self,
        axis: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> Result<Self::Output, TensorError>
    where
        O: BorrowMut<Self::Output>;
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
    // #[track_caller]
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
    #[track_caller]
    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool)
        -> Result<Self::Output, TensorError>;

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
    // #[track_caller]
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
    #[track_caller]
    fn mean<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

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
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;

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
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;

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
    #[track_caller]
    fn logsumexp<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;
}
