use std::borrow::BorrowMut;

use hpt_common::{axis::axis::Axis, error::base::TensorError};

/// A trait typically for argmax and argmin functions.
pub trait IndexReduce
where
    Self: Sized,
{
    /// The output tensor type.
    type Output;

    /// Return the indices of the maximum values along the specified dimensions
    ///
    /// ## Parameters:
    /// `dim`: Dimension to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0, 3.0, 2.0]);
    /// let r = a.argmax(0, true)?; // [0]
    /// ```
    #[track_caller]
    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Return the indices of the minimum values along the specified dimensions
    ///
    /// ## Parameters:
    /// `dim`: Dimension to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0, 3.0, 2.0]);
    /// let r = a.argmin(0, true)?; // [2]
    /// ```
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
    /// Compute the sum of elements along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.sum([0, 1], true)?; // [[10.]]
    /// ```
    #[track_caller]
    fn sum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Compute the sum of elements along the specified dimensions with specified output tensor
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// `init_out`: init the out data before doing computation   
    ///
    /// `out`: The output tensor
    ///
    /// ## Example:
    /// ```rust
    /// let d = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let e = Tensor::<f32>::new([[0.0]]);
    /// let f = d.sum_([0, 1], true, false, &mut e.clone())?; // [[10.]]
    /// ```
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

    /// Compute the product of elements along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.prod([0, 1], true)?; // [[24.]]
    /// ```
    #[track_caller]
    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Find the minimum of element along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.min([0, 1], true)?; // [[1.]]
    /// ```
    #[track_caller]
    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Find the maximum of element along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.max([0, 1], true)?; // [[4.]]
    /// ```
    #[track_caller]
    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Compute the L1 norm along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[-1.0, 2.0], [-3.0, 4.0]]);
    /// let d = c.reducel1([0, 1], true)?; // |-1| + |2| + |-3| + |4| = 10
    /// ```
    #[track_caller]
    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;

    /// Compute the sum of squares of elements along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.sum_square([0, 1], true)?; // [[30.]]  // 1^2 + 2^2 + 3^2 + 4^2 = 30
    /// ```
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
    /// Test if all elements are true along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.all([0, 1], true)?; // [[true]]
    /// ```
    #[track_caller]
    fn all<S: Into<Axis>>(&self, axis: S, keep_dims: bool)
        -> Result<Self::BoolOutput, TensorError>;

    /// Test if any elements are true along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.any([0, 1], true)?; // [[true]]
    /// ```
    #[track_caller]
    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool)
        -> Result<Self::BoolOutput, TensorError>;
}

/// A trait for tensor reduction operations, the output must remain the same tensor type.
pub trait NormalEvalReduce<T> {
    /// the output tensor type.
    type Output;
    /// Compute the sum of elements along the specified dimensions, treating NaNs as zero
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, f32::NAN], [3.0, 4.0]]);
    /// let d = c.nansum([0, 1], true)?; // [[8.]]
    /// ```
    #[track_caller]
    fn nansum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Compute the sum of elements along the specified dimensions, treating NaNs as zero with out with specified output tensor
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// `init_out`: init the out data before doing computation   
    ///
    /// `out`: The output tensor
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([1.0, f32::NAN, 3.0]);
    /// let mut out = Tensor::<f32>::new([0.0]);
    /// let mut b = a.nansum_([0], false, false, &mut out)?; // [4.]
    /// ```
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

    /// Compute the product of elements along the specified dimensions, treating NaNs as one
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, f32::NAN], [3.0, 4.0]]);
    /// let d = c.nanprod([0, 1], true)?; // [[12.]]
    /// ```
    #[track_caller]
    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool)
        -> Result<Self::Output, TensorError>;
}

/// A trait for tensor reduction operations, the output must be a floating-point tensor.
pub trait FloatReduce<T>
where
    Self: Sized,
{
    /// The output tensor type.
    type Output;

    /// Compute the mean of elements along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.mean([0, 1], true)?; // [[2.5]]
    /// ```
    #[track_caller]
    fn mean<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError>;

    /// Compute the L2 norm (Euclidean norm) along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [2.0, 3.0]]);
    /// let d = c.reducel2([0, 1], true)?; // sqrt(1^2 + 2^2 + 2^2 + 3^2) ≈ 4.24
    /// ```
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;

    /// Compute the L3 norm (Euclidean norm) along the specified dimensions
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [2.0, 3.0]]);
    /// let d = c.reducel3([0, 1], true)?; // [[3.5303]]
    /// ```
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;

    /// Compute `log(sum(exp(x_i)))` along the specified dimensions.
    ///
    /// ## Parameters:
    /// `dims`: Dimensions to reduce over
    ///
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = c.logsumexp([0, 1], true)?; // [[4.4401898]]
    /// ```
    #[track_caller]
    fn logsumexp<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Self::Output, TensorError>;
}
