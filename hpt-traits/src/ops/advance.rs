use hpt_common::error::base::TensorError;

/// A trait contains advance operations
pub trait AdvancedOps {
    /// The type of the meta data
    type Meta;
    /// The type of the output tensor
    type Output;
    /// The type of the index tensor
    type IndexOutput;
    /// Pads a tensor with a given constant value. For each dimension, adds padding at the start and end as specified by `pads`.
    ///
    /// ## Parameters:
    /// `pads`: A slice of tuples where each tuple contains two values (before_pad, after_pad) for each dimension. The length must match the number of dimensions in the input tensor.
    ///
    /// `val`: The constant value to use for padding.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]); // 2x2 matrix
    /// let pads = &[(1, 0), (0, 2)];
    /// let result = x.pad(pads, 0.0)?;
    /// ```
    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError>;
    /// Returns the k largest or smallest elements along a specified dimension, and their indices.
    ///
    /// ## Parameters:
    /// `k`: Number of top elements to return.
    ///
    /// `dim`: The dimension to sort along. Supports negative indexing.
    ///
    /// `largest`: If true, returns the k largest elements; if false, the k smallest.
    ///
    /// `sorted`: If true, the returned elements are sorted in descending/ascending order.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f64>::new(&[5., 2., 8., 1., 9., 3.]);
    ///
    /// // Get top 3 largest values and their indices
    /// let (indices, values) = x.topk(3, 0, true, true)?;
    /// println!("Top 3 values: {}", values); // [9., 8., 5.]
    /// println!("Their indices: {}", indices); // [4, 2, 0]
    ///
    /// // Get top 2 smallest values, unsorted
    /// let (indices, values) = x.topk(2, 0, false, false)?;
    /// println!("Bottom 2 values: {}", values); // Values might be in any order
    /// ```
    fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self::IndexOutput, Self::Output), TensorError>;
    /// Creates a one-hot tensor from the input tensor.
    ///
    /// The output tensor will have an additional dimension of size `depth` inserted at `axis`,
    /// where indices from the input tensor select which index in this dimension gets the `true_val` value while all other indices get the `false_val` value.
    ///
    /// ## Parameters:
    /// `depth`: Size of the new dimension.
    ///
    /// `axis`: Position to insert the new dimension. Supports negative indexing.
    ///
    /// `true_val`: Value to place at the index specified by the input tensor.
    ///
    /// `false_val`: Value to place at all other indices.
    ///
    /// ## Example:
    /// ```rust
    /// let indices = Tensor::<i64>::new(&[1, 0, 2]);
    /// // Create one-hot encoding with depth 3
    /// let onehot = indices.onehot(3, -1, 1, 0)?;
    /// println!("One-hot encoding:\n{}", onehot);
    /// // Output:
    /// // [[0., 1., 0.],
    /// //  [1., 0., 0.],
    /// //  [0., 0., 1.]]
    /// ```
    fn onehot(
        &self,
        depth: usize,
        axis: i64,
        true_val: Self::Meta,
        false_val: Self::Meta,
    ) -> Result<Self::Output, TensorError>;
    /// Writes values from `src` tensor into a new tensor at the indices specified by `indices` along dimension `axis`.
    ///
    /// The rest of the values in the output tensor are copied from the input tensor `x`.
    ///
    /// ## Parameters:
    /// `indices`: Index tensor that specifies where to scatter the values.
    ///
    /// `axis`: The axis along which to scatter values. Supports negative indexing.
    ///
    /// `src`: The tensor containing values to scatter.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f64>::zeros(&[3, 5])?; // base tensor
    /// let src = Tensor::<f64>::new(&[1., 2., 3.]);
    /// let indices = Tensor::<i64>::new(&[0, 2, 4]);
    /// let result = x.scatter(&indices, /*dim =*/1, &src)?;
    /// println!("Result:\n{}", result);
    /// // Output:
    /// // [[1. 0. 2. 0. 3.],
    /// //  [0. 0. 0. 0. 0.],
    /// //  [0. 0. 0. 0. 0.]]
    /// ```
    fn scatter(
        &self,
        indices: &Self::IndexOutput,
        axis: i64,
        src: &Self::Output,
    ) -> Result<Self::Output, TensorError>;
}

/// A trait for hardmax
pub trait HardMax<T> {
    /// The type of the output tensor
    type Output;
    /// Applies the hardmax function to the input tensor along the specified dimension.
    ///
    /// The hardmax function sets the largest element along the specified dimension to 1 and all other elements to 0.
    ///
    /// ## Parameters:
    /// `axis`: The dimension along which to apply the hardmax.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::new(&[[-1.0, 0.0, 3.0], [2.0, 1.0, 4.0]]);
    /// let result = x.hardmax(/*dim =*/1)?;
    /// println!("Result:\n{}", result);
    /// // Output:
    /// // [[0, 0, 1],
    /// //  [0, 0, 1]]
    /// ```
    fn hardmax(&self, axis: i64) -> Result<Self::Output, TensorError>;
}

/// A trait for tensor where
pub trait TensorWhere {
    /// The type of the output tensor
    type Output;
    /// The type of the condition tensor
    type Condition;
    /// Element-wise selection based on a condition tensor. Returns a tensor of elements selected from `x` where condition is true, and from `y` where condition is false.
    ///
    /// ## Parameters:
    /// `condition`: A boolean tensor that determines which elements to select.
    ///
    /// `x`: Tensor whose elements are selected where condition is true.
    ///
    /// `y`: Tensor whose elements are selected where condition is false.
    ///
    /// ## Example:
    /// ```rust
    /// let condition = Tensor::<bool>::new(&[true, false, true]);
    /// let x = Tensor::<f64>::new(&[1., 2., 3.]);
    /// let y = Tensor::<f64>::new(&[4., 5., 6.]);
    /// let result = Tensor::tensor_where(&condition, &x, &y)?;
    /// println!("{}", result); // [1., 5., 3.]
    /// ```
    fn tensor_where(
        condition: &Self::Condition,
        x: &Self::Output,
        y: &Self::Output,
    ) -> Result<Self::Output, TensorError>;
}
