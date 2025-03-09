use hpt_common::{axis::axis::Axis, error::base::TensorError, shape::shape::Shape};

/// A trait for manipulating the shape of a tensor.
pub trait ShapeManipulate
where
    Self: Sized,
{
    /// tensor data type
    type Meta;
    /// the output type
    type Output;

    /// Remove single-dimensional entries (axes with size 1) from the shape of the tensor at specified positions.
    ///
    /// ## Parameters:
    /// `axes`: The positions where the single-dimensional entries should be removed.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[1, 3, 1, 4])?;
    /// let b = a.squeeze(0)?; // shape becomes [3, 1, 4]
    /// let c = a.squeeze(2)?; // shape becomes [1, 3, 4]
    /// ```
    #[track_caller]
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError>;

    /// Adds a new dimension of size 1 to the tensor at the specified dimention.
    ///
    /// ## Parameters:
    /// `axes`: The positions where the single-dimensional entries should be add.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[3, 4])?;
    /// let b = a.unsqueeze(0)?; // shape becomes [1, 3, 4]
    /// let c = a.unsqueeze(1)?; // shape becomes [3, 1, 4]
    /// ```
    #[track_caller]
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError>;

    /// Gives a new shape to the tensor without changing its data when it is possible.
    ///
    /// ## Parameters:
    /// `shape`: The new shape. The total number of elements must remain the same.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[3, 4])?;
    /// let b = a.reshape(&[2, 6])?;
    /// let c = a.reshape(&[12])?;
    /// ```
    #[track_caller]
    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, TensorError>;

    /// Swaps two axes of the 2D tensor, returning a view of the tensor with axes transposed.
    ///
    /// ## Parameters:
    /// `axis1`: First axis to be transposed
    ///
    /// `axis2`: Second axis to be transposed
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[2, 4])?;
    /// let b = a.transpose(0, 1)?; // shape becomes [2, 4]
    /// let c = a.transpose(1, 0)?; // shape becomes [4, 2]
    /// ```
    #[track_caller]
    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, TensorError>;

    /// Permutes the dimensions of the tensor according to the given axes order.
    ///
    /// ## Parameters:
    /// `axes`: The desired ordering of dimensions. Must be a permutation of [0, 1, ..., n-1] where n is the number of dimensions.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[2, 3, 4])?;
    /// let b = a.permute(&[2, 0, 1])?; // Permute dimensions to [4, 2, 3]
    /// let c = a.permute(&[1, 2, 0])?; // Permute dimensions to [3, 4, 2]
    /// let d = a.permute(&[1, 1, 0]); // This will return an error as [1, 1, 0] is not a valid permutation
    /// ```
    #[track_caller]
    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError>;

    /// Performs the inverse permutation of dimensions according to the given axes order. This is equivalent to undoing a previous permutation.
    ///
    /// ## Parameters:
    /// `axes`: The permutation to invert. Must be a permutation of [0, 1, ..., n-1] where n is the number of dimensions.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[2, 3, 4])?;
    /// let b = a.permute(&[2, 0, 1])?; // Permute dimensions to [4, 2, 3]
    /// let c = b.permute_inv(&[2, 0, 1])?; // Apply inverse permutation to get back original shape
    /// ```
    #[track_caller]
    fn permute_inv<A: Into<Axis>>(&self, axes: A)
        -> std::result::Result<Self::Output, TensorError>;

    /// Expands the tensor to a larger size, replicating the data along specified dimensions.
    ///
    /// ## Parameters:
    /// `shape`: The desired expanded shape. Must be compatible with the input tensor's shape, where each dimension must either be equal to the input dimension or the input dimension must be 1.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[1, 3, 1])?;
    /// let b = a.expand(&[2, 3, 4])?;
    /// ```
    #[track_caller]
    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, TensorError>;

    /// Transposes the tensor by swapping the last two dimensions. For 1D or 2D tensors, this is equivalent to a regular transpose. For higher dimensional tensors, only the last two dimensions are swapped.
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::zeros(&[2, 3, 4])?;
    /// let d = c.t()?; // shape becomes [2, 4, 3]
    /// ```
    #[track_caller]
    fn t(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Performs a complete transpose by reversing all dimensions of the tensor. This is different from `t()` which only swaps the last two dimensions.
    ///
    /// ## Example:
    /// ```rust
    /// let c = Tensor::<f32>::zeros(&[2, 3, 4])?;
    /// let d = c.mt()?; // shape becomes [4, 3, 2]
    /// ```
    #[track_caller]
    fn mt(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Reverses the order of elements in the tensor along the specified axes.
    ///
    /// ## Parameters:
    /// `axes`: The axes along which to flip the tensor. Can be a single axis or multiple axes.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    /// let b = a.flip(0)?;
    /// // [[4, 5, 6],
    /// //  [1, 2, 3]]
    /// ```
    #[track_caller]
    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError>;

    /// Reverses the order of elements along axis 1 (columns) of the tensor. The tensor must be at least 2-dimensional.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    /// let b = a.fliplr()?;
    /// // [[3, 2, 1],
    /// //  [6, 5, 4]]
    /// ```
    #[track_caller]
    fn fliplr(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Reverses the order of elements along axis 0 (rows) of the tensor. The tensor must be at least 1-dimensional.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    /// let b = a.flipud()?;
    /// // [[4, 5, 6],
    /// //  [1, 2, 3]]
    /// ```
    #[track_caller]
    fn flipud(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Constructs a new tensor by repeating the input tensor along specified dimensions.
    ///
    /// ## Parameters:
    /// `repeats`: The number of repetitions for each dimension.
    /// If `repeats` has fewer dimensions than the input tensor, it is padded with 1s. If `repeats` has more dimensions than the input tensor, the input tensor is padded with dimensions of size 1.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    /// // [[1, 2],
    /// //  [3, 4]]
    /// let b = a.tile(&[2, 1])?;
    /// // [[1, 2],
    /// //  [3, 4],
    /// //  [1, 2],
    /// //  [3, 4]]
    /// ```
    #[track_caller]
    fn tile<S: Into<Axis>>(&self, repeats: S) -> std::result::Result<Self::Output, TensorError>;

    /// Removes zeros from the beginning and/or end of a 1-D tensor.
    ///
    /// ## Parameters:
    /// `trim`: A string specifying which zeros to remove:
    /// - 'f': remove leading zeros (from front)
    /// - 'b': remove trailing zeros (from back)
    /// - 'fb' or 'bf': remove both leading and trailing zeros
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    /// let b = a.trim_zeros("f")?; // [1, 2, 3, 0, 0]
    /// let c = a.trim_zeros("b")?; // [0, 0, 1, 2, 3]
    /// let d = a.trim_zeros("fb")?; // [1, 2, 3]
    /// ```
    #[track_caller]
    fn trim_zeros(&self, trim: &str) -> std::result::Result<Self::Output, TensorError>
    where
        Self::Meta: PartialEq;

    /// Repeats elements of a tensor along a specified axis.
    ///
    /// ## Parameters:
    /// `repeats`: Number of repetitions for each element
    ///
    /// `axis`: The axis along which to repeat values. Negative values count from the end
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    /// // [[1, 2],
    /// //  [3, 4]]
    /// let b = a.repeat(2, 0)?;
    /// // [[1, 2],
    /// //  [1, 2],
    /// //  [3, 4],
    /// //  [3, 4]]
    /// ```
    #[track_caller]
    fn repeat(&self, repeats: usize, axis: i16) -> std::result::Result<Self::Output, TensorError>;

    /// Splits a tensor into multiple sub-tensors along a specified axis at given indices.
    ///
    /// ## Parameters:
    /// `indices_or_sections`: The indices where the splits should occur
    ///
    /// `axis`: The axis along which to split the tensor. Negative values count from the end
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let splits = a.split(&[2, 4], 0)?;
    /// // splits[0]: [1, 2]
    /// // splits[1]: [3, 4]
    /// // splits[2]: [5, 6]
    /// ```
    #[track_caller]
    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> std::result::Result<Vec<Self::Output>, TensorError>;

    /// Splits a tensor into multiple sub-tensors along axis 2 (depth). The tensor must be at least 3-dimensional.
    ///
    /// ## Parameters:
    /// `indices`: The indices where the splits should occur along axis 2
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0,
    ///                             5.0, 6.0, 7.0, 8.0,
    ///                             9.0, 10.0, 11.0, 12.0,
    ///                             13.0, 14.0, 15.0, 16.0]).reshape(&[2, 2, 4])?;
    /// let splits = a.dsplit(&[2])?;
    /// // splits[0]: shape [2, 2, 2]
    /// // [[[1, 2],
    /// //   [5, 6]],
    /// //  [[9, 10],
    /// //   [13, 14]]]
    /// // splits[1]: shape [2, 2, 2]
    /// // [[[3, 4],
    /// //   [7, 8]],
    /// //  [[11, 12],
    /// //   [15, 16]]]
    /// ```
    #[track_caller]
    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self::Output>, TensorError>;

    /// Splits a tensor into multiple sub-tensors horizontally (along axis 1). The tensor must be at least 2-dimensional.
    ///
    /// ## Parameters:
    /// `indices`: The indices where the splits should occur along axis 1 (columns)
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0,
    ///                             5.0, 6.0, 7.0, 8.0]).reshape(&[2, 4])?;
    /// let splits = a.hsplit(&[2])?;
    /// // splits[0]:
    /// // [[1, 2],
    /// //  [5, 6]]
    /// // splits[1]:
    /// // [[3, 4],
    /// //  [7, 8]]
    /// ```
    #[track_caller]
    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self::Output>, TensorError>;

    /// Splits a tensor into multiple sub-tensors vertically (along axis 0). The tensor must be at least 1-dimensional.
    ///
    /// ## Parameters:
    /// `indices`: The indices where the splits should occur along axis 0 (rows)
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0,
    ///                             3.0, 4.0,
    ///                             5.0, 6.0,
    ///                             7.0, 8.0]).reshape(&[4, 2])?;
    /// let splits = a.vsplit(&[2])?;
    /// // splits[0]:
    /// // [[1, 2],
    /// //  [3, 4]]
    /// // splits[1]:
    /// // [[5, 6],
    /// //  [7, 8]]
    /// ```
    #[track_caller]
    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self::Output>, TensorError>;

    /// Interchanges two axes of a tensor. This operation creates a view of the tensor with the specified axes swapped.
    ///
    /// ## Parameters:
    /// `axis1`: First axis to be swapped
    ///
    /// `axis2`: Second axis to be swapped
    ///
    /// Both axes can be negative, counting from the end of the dimensions.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    /// let b = a.swap_axes(0, 1)?;
    /// ```
    #[track_caller]
    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, TensorError>;

    /// Flattens a contiguous range of dimensions in a tensor into a single dimension.
    ///
    /// ## Parameters:
    /// `start_dim`: Starting dimension to flatten (inclusive). Defaults to 0 if None
    ///
    /// `end_dim`: Ending dimension to flatten (inclusive). Defaults to last dimension if None
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ///                             7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[2, 3, 2])?;
    /// let b = a.flatten(None, None)?; // Flatten all dimensions (default behavior)
    /// let c = a.flatten(Some(1), Some(2))?; // Shape: [2, 6]
    /// ```
    #[track_caller]
    fn flatten<A>(&self, start: A, end: A) -> std::result::Result<Self::Output, TensorError>
    where
        A: Into<Option<usize>>;
}

/// trait for concat
pub trait Concat: Sized {
    /// the output type of concat
    type Output;
    /// Concatenates a sequence of tensors along the specified axis.
    ///
    /// ## Parameters:
    /// `tensors`: Vector of tensors to concatenate
    ///
    /// `axis`: The axis along which to concatenate the tensors
    ///
    /// `keepdims`: If true, inserts a new dimension at the concatenation axis, splitting the concatenated dimension into [num_tensors, concatenated_size]
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    /// let b = Tensor::<f32>::new(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2])?;
    /// let e = Tensor::concat(vec![a.clone(), b.clone()], 0, true)?; // Shape: [2, 2, 2]
    /// ```
    #[track_caller]
    fn concat(
        tensors: Vec<Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self::Output, TensorError>;

    /// Stacks tensors vertically (along axis 0). This is equivalent to concatenation along the first axis.
    ///
    /// ## Parameters:
    /// `tensors`: Vector of tensors to stack vertically
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[1, 3])?;
    /// let b = Tensor::<f32>::new(&[4.0, 5.0, 6.0]).reshape(&[1, 3])?;
    /// let c = Tensor::vstack(vec![a.clone(), b.clone()])?;
    /// ```
    #[track_caller]
    fn vstack(tensors: Vec<Self>) -> std::result::Result<Self::Output, TensorError>;

    /// Stacks tensors horizontally (along axis 1 for 2D+ tensors, or axis 0 for 1D tensors).
    ///
    /// ## Parameters:
    /// `tensors`: Vector of tensors to stack horizontally
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    /// let b = Tensor::<f32>::new(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2])?;
    /// let c = Tensor::hstack(vec![a.clone(), b.clone()])?;
    /// ```
    #[track_caller]
    fn hstack(tensors: Vec<Self>) -> std::result::Result<Self::Output, TensorError>;

    /// Stacks tensors along the third axis (depth). Input tensors are promoted to 3D if necessary.
    ///
    /// ## Parameters:
    /// `tensors`: Vector of tensors to stack along depth
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2, 1])?;
    /// let b = Tensor::<f32>::new(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2, 1])?;
    /// let c = Tensor::dstack(vec![a.clone(), b.clone()])?;
    /// ```
    #[track_caller]
    fn dstack(tensors: Vec<Self>) -> std::result::Result<Self::Output, TensorError>;
}
