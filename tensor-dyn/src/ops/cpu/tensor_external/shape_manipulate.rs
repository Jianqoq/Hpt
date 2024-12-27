use crate::ops::cpu::concat::concat;
use crate::{ tensor::Tensor, tensor_base::_Tensor };
use tensor_common::err_handler::ErrHandler;
use tensor_common::{ axis::Axis, shape::Shape };
use tensor_traits::{ CommonBounds, ShapeManipulate };

impl<T: CommonBounds> ShapeManipulate for Tensor<T> {
    type Meta = T;
    type Output = Tensor<T>;

    /// Removes dimensions of size 1 from the tensor along the specified axes.
    ///
    /// this operation reduces the dimensionality of the tensor by "squeezing"
    /// out the dimensions that have a size of 1. If `axes` are specified, only those axes will be squeezed.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axis or axes to squeeze. This can be a single axis or a set of axes where the dimension
    /// should be removed, and the dimension at each axis must be of size 1. The type `A` must implement the `Into<Axis>` trait.
    ///
    /// # Returns
    ///
    /// * If the operation is successful, it returns the tensor with the specified dimensions removed. If no axes
    /// are specified, all axes of size 1 will be removed.
    ///
    /// # Panics
    ///
    /// * This function will panic if the specified axes do not have a dimension of 1 or if the axes are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[1, 3, 1]).unwrap();
    /// let squeezed_tensor = tensor.squeeze(0).unwrap();
    /// assert_eq!(squeezed_tensor.shape().inner(), &[3]);
    /// ```
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::squeeze(self.inner.as_ref(), axes)?.into())
    }

    /// Adds a new dimension of size 1 to the tensor at the specified axes.
    ///
    /// this operation increases the dimensionality of the tensor by "unsqueezing"
    /// and introducing new axes of size 1 at the given positions. This is often used to reshape
    /// tensors for operations that require specific dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axis or axes at which to add the new dimension. The type `A` must implement the `Into<Axis>` trait.
    /// Each specified axis must be a valid index in the tensor, and after the operation, the tensor will have a dimension of 1 at those positions.
    ///
    /// # Returns
    ///
    /// * Returns the tensor with the new dimensions of size 1 added at the specified axes.
    ///
    /// # Panics
    ///
    /// * This function will panic if the specified axes are out of bounds for the tensor.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[3]).unwrap();
    /// let unsqueezed_tensor = tensor.unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed_tensor.shape().inner(), &[1, 3]);
    /// ```
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::unsqueeze(self.inner.as_ref(), axes)?.into())
    }

    /// Reshapes the tensor into the specified shape without changing its data.
    ///
    /// this operation rearranges the elements of the tensor into a new shape,
    /// as long as the total number of elements remains the same.
    /// The operation is performed without copying or modifying the underlying data only when the
    /// dimension to manipulate is contiguous, otherwise, a new Tensor will return.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the tensor. The type `S` must implement the `Into<Shape>` trait.
    /// The total number of elements in the new shape must match the total number of elements in the original tensor.
    ///
    /// # Returns
    ///
    /// * Returns the reshaped tensor if the operation is successful.
    ///
    /// # Panics
    ///
    /// * This function will panic if the new shape does not match the total number of elements in the original tensor.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[3]).unwrap();
    /// let reshaped_tensor = tensor.reshape(&[1, 3]).unwrap();
    /// assert_eq!(reshaped_tensor.shape().inner(), &[1, 3]);
    /// ```
    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::reshape(self.inner.as_ref(), shape)?.into())
    }

    /// Swaps two axes of the tensor, effectively transposing the dimensions along the specified axes.
    ///
    /// this operation switches the data between the two specified axes, changing the layout
    /// of the tensor without altering its data. Transposing is commonly used to change the orientation
    /// of matrices or tensors.
    ///
    /// # Arguments
    ///
    /// * `axis1` - The first axis to be swapped. Must be a valid axis index within the tensor.
    /// * `axis2` - The second axis to be swapped. Must also be a valid axis index within the tensor.
    ///
    /// # Returns
    ///
    /// * Returns the tensor with the two specified axes transposed.
    ///
    /// # Panics
    ///
    /// * This function will panic if either `axis1` or `axis2` are out of bounds or if they are not valid dimensions of the tensor.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]).unwrap();
    /// let transposed_tensor = tensor.transpose(0, 1).unwrap();
    /// assert_eq!(transposed_tensor.shape().inner(), &[3, 1]);
    /// ```
    ///
    /// # See Also
    /// - [`permute`]: Rearranges all axes of the tensor according to a given order.
    /// - [`swap_axes`]: Swaps two specified axes in the tensor (an alias for `transpose`).
    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::transpose(self.inner.as_ref(), axis1, axis2)?.into())
    }

    /// Reorders the dimensions of the tensor according to the specified axes.
    ///
    /// This operation permutes the dimensions of the tensor based on the provided axes,
    /// effectively changing the layout of the data. Each axis in the tensor is rearranged to follow the
    /// order specified in the `axes` argument, allowing for flexible reordering of the tensor’s dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - A list or sequence of axes that specifies the new order of the dimensions. The type `A` must implement
    /// the `Into<Axis>` trait. The length of `axes` must match the number of dimensions in the tensor.
    ///
    /// # Returns
    ///
    /// * The tensor with its dimensions permuted according to the specified `axes`.
    ///
    /// # Panics
    ///
    /// * This function will panic if the length of `axes` does not match the number of dimensions in the tensor,
    /// or if any of the axes are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]).unwrap();
    /// let permuted_tensor = tensor.permute(&[1, 0]).unwrap();
    /// assert_eq!(permuted_tensor.shape().inner(), &[3, 1]);
    /// ```
    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::permute(self.inner.as_ref(), axes)?.into())
    }

    /// Reverses the permutation of the dimensions of the tensor according to the specified axes.
    ///
    /// This operation is the inverse of the `permute` function. It restores the original order of the dimensions
    /// based on the provided axes, effectively undoing a previous permutation.
    ///
    /// # Arguments
    ///
    /// * `axes` - A list or sequence of axes that specifies the inverse order to restore the original layout of the tensor.
    /// The type `A` must implement the `Into<Axis>` trait. The length of `axes` must match the number of dimensions in the tensor.
    ///
    /// # Returns
    ///
    /// * The tensor with its dimensions restored to their original order based on the inverse of the specified `axes`.
    ///
    /// # Panics
    ///
    /// * This function will panic if the length of `axes` does not match the number of dimensions in the tensor,
    /// or if any of the axes are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]).unwrap();
    /// let permuted_tensor = tensor.permute(&[1, 0]).unwrap();
    /// let restored_tensor = permuted_tensor.permute_inv(&[1, 0]).unwrap();
    /// assert_eq!(restored_tensor.shape().inner(), &[1, 3]);
    /// ```
    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::permute_inv(self.inner.as_ref(), axes)?.into())
    }

    /// Expands the tensor to a larger shape without copying data, using broadcasting.
    ///
    /// This operation expands the dimensions of the tensor according to the specified shape,
    /// using broadcasting rules. The expanded tensor will have the same data as the original, but with
    /// new dimensions where the size of 1 can be expanded to match the target shape. No new memory
    /// is allocated for the expanded dimensions.
    ///
    /// # Arguments
    ///
    /// * `shape` - The target shape to expand the tensor to. The type `S` must implement the `Into<Shape>` trait.
    /// The specified shape must be compatible with the current shape, following broadcasting rules (i.e., existing
    /// dimensions of size 1 can be expanded to larger sizes).
    ///
    /// # Returns
    ///
    /// * The tensor expanded to the specified shape using broadcasting.
    ///
    /// # Panics
    ///
    /// * This function will panic if the target shape is incompatible with the tensor's current shape,
    /// or if the dimension to expand is not `1`.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]).unwrap();
    /// let expanded_tensor = tensor.expand(&[2, 3]).unwrap();
    /// assert_eq!(expanded_tensor.shape().inner(), &[2, 3]);
    /// ```
    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::expand(self.inner.as_ref(), shape)?.into())
    }

    /// Returns the transpose of the tensor by swapping the last two dimensions.
    ///
    /// This operation is typically used for 2D tensors (matrices) but can also be applied to higher-dimensional tensors,
    /// where it swaps the last two dimensions. It is a shorthand for a specific case of transposition that is often used in
    /// linear algebra operations.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * The tensor with its last two dimensions transposed.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor has fewer than two dimensions, as transposing the last two dimensions requires
    /// the tensor to be at least 2D.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let transposed_tensor = tensor.t().unwrap();
    /// assert_eq!(transposed_tensor.shape().inner(), &[2, 2]);
    /// ```
    fn t(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::t(self.inner.as_ref())?.into())
    }

    /// reverse the dimensions of the tensor.
    ///
    /// This operation transposes a N-dimensional tensor by reversing the order of the dimensions.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * The transposed matrix (2D tensor) where the rows and columns have been swapped.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor is not 2D. It is only valid for matrices.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let transposed_tensor = tensor.mt().unwrap();
    /// assert_eq!(transposed_tensor.shape().inner(), &[2, 2]);
    /// ```
    fn mt(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::mt(self.inner.as_ref())?.into())
    }

    /// Reverses the order of elements along the specified axes of the tensor.
    ///
    /// This operation flips the tensor along the given axes, effectively reversing the order of elements
    /// along those dimensions. The rest of the tensor remains unchanged.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axis or axes along which to flip the tensor. The type `A` must implement the `Into<Axis>` trait.
    ///
    /// # Returns
    ///
    /// * The tensor with elements reversed along the specified axes.
    ///
    /// # Panics
    ///
    /// * This function will panic if any of the specified axes are out of bounds for the tensor.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0])
    ///     .reshape(&[2, 2])
    ///     .unwrap();
    /// let flipped_tensor = tensor.flip(0).unwrap();
    /// assert!(flipped_tensor.allclose(
    ///     &Tensor::<f64>::new(vec![3.0, 4.0, 1.0, 2.0])
    ///         .reshape(&[2, 2])
    ///         .unwrap()
    /// ));
    /// ```
    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::flip(self.inner.as_ref(), axes)?.into())
    }

    /// Reverses the order of elements along the last dimension (columns) of a 2D tensor (matrix).
    ///
    /// This operation flips the tensor from left to right, effectively reversing the order of the elements
    /// in each row for 2D tensors. For higher-dimensional tensors, this function flips the elements along
    /// the second-to-last axis.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * The tensor with elements flipped along the last axis for 2D tensors, or along the second-to-last axis
    ///   for higher-dimensional tensors.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor has fewer than two dimensions.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0])
    ///     .reshape(&[2, 2])
    ///     .unwrap();
    /// let flipped_tensor = tensor.fliplr().unwrap();
    /// assert!(flipped_tensor.allclose(
    ///     &Tensor::<f64>::new(vec![2.0, 1.0, 4.0, 3.0])
    ///         .reshape(&[2, 2])
    ///         .unwrap()
    /// ));
    /// ```
    fn fliplr(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::fliplr(self.inner.as_ref())?.into())
    }

    /// Reverses the order of elements along the first dimension (rows) of a 2D tensor (matrix).
    ///
    /// This operation flips the tensor upside down, effectively reversing the order of the rows in a 2D tensor.
    /// For higher-dimensional tensors, this function flips the elements along the first axis.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * The tensor with elements flipped along the first axis (rows for 2D tensors).
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor has fewer than one dimension.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0])
    ///     .reshape(&[2, 2])
    ///     .unwrap();
    /// let flipped_tensor = tensor.flipud().unwrap();
    /// assert!(flipped_tensor.allclose(
    ///     &Tensor::<f64>::new(vec![3.0, 4.0, 1.0, 2.0])
    ///         .reshape(&[2, 2])
    ///         .unwrap()
    /// ));
    /// ```
    fn flipud(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::flipud(self.inner.as_ref())?.into())
    }

    /// Repeats the tensor along the specified axes according to the given repetition values.
    ///
    /// This operation tiles (repeats) the tensor along each axis a specified number of times, effectively
    /// creating a larger tensor by replicating the original tensor's data. Each dimension is repeated according
    /// to the corresponding value in `reps`.
    ///
    /// # Arguments
    ///
    /// * `reps` - A sequence that specifies how many times to repeat the tensor along each axis. The type `S` must
    ///   implement the `Into<Axis>` trait. The length of `reps` must match the number of dimensions in the tensor.
    ///
    /// # Returns
    ///
    /// * The tensor with its elements repeated along the specified axes, resulting in a tiled version of the original tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the length of `reps` does not match the number of dimensions in the tensor,
    ///   or if any repetition value is negative.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///    .reshape(&[1, 3])
    ///   .unwrap();
    /// let repeated_tensor = tensor.tile(&[2, 1]).unwrap();
    /// assert!(repeated_tensor.allclose(
    ///    &Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    ///       .reshape(&[2, 3])
    ///      .unwrap()
    /// ));
    /// ```
    fn tile<S: Into<Axis>>(&self, repeats: S) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::tile(self.inner.as_ref(), repeats)?.into())
    }

    /// Removes leading or trailing zeros from the tensor based on the specified trim mode.
    ///
    /// This operation trims the tensor by removing zeros from either the start or the end of the tensor,
    /// or both, depending on the `trim` argument. It is useful for cleaning up tensors where unnecessary
    /// zero-padding exists.
    ///
    /// # Arguments
    ///
    /// * `trim` - A string that specifies how to trim the zeros. Accepted values are:
    ///   - `"leading"`: Removes zeros from the start of the tensor.
    ///   - `"trailing"`: Removes zeros from the end of the tensor.
    ///   - `"both"`: Removes zeros from both the start and the end of the tensor.
    ///
    /// # Returns
    ///
    /// * The tensor with zeros removed from the specified locations (leading, trailing, or both).
    ///
    /// # Panics
    ///
    /// * This function will panic if `trim` is not one of the accepted values (`"leading"`, `"trailing"`, or `"both"`).
    ///
    /// # Requirements
    ///
    /// * `Self::Meta` must implement `PartialEq`, as it is used to compare the tensor elements for equality with zero.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
    ///     .reshape(&[6])
    ///     .unwrap();
    /// let trimmed_tensor = tensor.trim_zeros("fb").unwrap();
    /// assert!(trimmed_tensor.allclose(&Tensor::<f64>::new(vec![1.0, 2.0]).reshape(&[2]).unwrap()));
    /// ```
    fn trim_zeros(&self, trim: &str) -> std::result::Result<Self::Output, ErrHandler> where Self::Meta: PartialEq {
        Ok(_Tensor::trim_zeros(self.inner.as_ref(), trim)?.into())
    }

    /// Repeats the elements of the tensor along the specified axis a given number of times.
    ///
    /// This operation creates a new tensor where the elements along the specified axis are repeated `repeats` times,
    /// effectively increasing the size of that axis while duplicating the data in the original tensor.
    ///
    /// # Arguments
    ///
    /// * `repeats` - The number of times to repeat each element along the specified axis.
    /// * `axis` - The axis along which to repeat the elements. Must be a valid axis index for the tensor.
    ///
    /// # Returns
    ///
    /// * A new tensor where the elements are repeated `repeats` times along the specified `axis`.
    ///
    /// # Panics
    ///
    /// * This function will panic if the `axis` is out of bounds for the tensor, or if `repeats` is zero.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///     .reshape(&[1, 3])
    ///     .unwrap();
    /// let repeated_tensor = tensor.repeat(2, 0).unwrap();
    /// assert!(repeated_tensor.allclose(
    ///     &Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    ///         .reshape(&[2, 3])
    ///         .unwrap()
    /// ));
    /// ```
    fn repeat(&self, repeats: usize, axes: i16) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::repeat(self.inner.as_ref(), repeats, axes)?.into())
    }

    /// Splits the tensor into multiple sub-tensors along the specified axis.
    ///
    /// This operation divides the tensor into smaller tensors according to the provided `indices_or_sections`.
    /// If `indices_or_sections` is a list of indices, the tensor is split at those indices along the given `axis`.
    /// If it is a single integer, the tensor is split into that many equal parts along the axis. The result is a
    /// vector of sub-tensors.
    ///
    /// # Arguments
    ///
    /// * `indices_or_sections` - A slice of indices or an integer. If it is a list of indices, the tensor is split at
    ///   each index along the specified axis. If it is an integer, the tensor is split into that many equal-sized sections.
    /// * `axis` - The axis along which to split the tensor. Must be a valid axis index for the tensor.
    ///
    /// # Returns
    ///
    /// * A vector of sub-tensors resulting from the split along the specified axis.
    ///
    /// # Panics
    ///
    /// * This function will panic if `axis` is out of bounds for the tensor.
    /// * It will also panic if the indices in `indices_or_sections` are out of range, or if `indices_or_sections` is an integer
    ///   that does not evenly divide the tensor along the specified axis.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[2, 3])
    ///     .unwrap();
    /// let split_tensors = tensor.split(&[1], 0).unwrap();
    /// assert!(split_tensors[0].allclose(
    ///     &Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///         .reshape(&[1, 3])
    ///         .unwrap()
    /// ));
    /// assert!(split_tensors[1].allclose(
    ///     &Tensor::<f64>::new(vec![4.0, 5.0, 6.0])
    ///         .reshape(&[1, 3])
    ///         .unwrap()
    /// ));
    /// ```
    fn split(&self, indices_or_sections: &[i64], axis: i64) -> std::result::Result<Vec<Self>, ErrHandler> {
        Ok(
            _Tensor
                ::split(self.inner.as_ref(), indices_or_sections, axis)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    /// Splits the tensor into multiple sub-tensors along the depth axis (third dimension).
    ///
    /// This function divides the tensor along its third dimension based on the provided `indices`.
    /// It is typically used for 3D tensors (or higher).
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of indices at which to split the tensor along the depth axis.
    ///
    /// # Returns
    ///
    /// * A list of sub-tensors split along the depth axis.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor has fewer than 3 dimensions or if any of the indices are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[1, 2, 3])
    ///     .unwrap();
    /// let split_tensors = tensor.dsplit(&[1]).unwrap();
    /// assert!(split_tensors[0].allclose(
    ///     &Tensor::<f64>::new(vec![1.0, 4.0])
    ///         .reshape(&[1, 2, 1])
    ///         .unwrap()
    /// ));
    /// assert!(split_tensors[1].allclose(
    ///     &Tensor::<f64>::new(vec![2.0, 3.0, 5.0, 6.0])
    ///         .reshape(&[1, 2, 2])
    ///         .unwrap()
    /// ));
    /// ```
    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, ErrHandler> {
        Ok(
            _Tensor
                ::dsplit(self.inner.as_ref(), indices)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    /// Splits the tensor into multiple sub-tensors along the horizontal axis (second dimension).
    ///
    /// This function divides the tensor along its second dimension (columns) based on the provided `indices`.
    /// It is typically used for 2D tensors (matrices) or higher.
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of indices at which to split the tensor along the horizontal axis.
    ///
    /// # Returns
    ///
    /// * A vector of sub-tensors split along the horizontal axis.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor has fewer than 2 dimensions or if any of the indices are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[1, 2, 3])
    ///     .unwrap();
    /// let split_tensors = tensor.hsplit(&[1]).unwrap();
    /// assert!(split_tensors[0].allclose(
    ///     &Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///         .reshape(&[1, 1, 3])
    ///         .unwrap()
    /// ));
    /// assert!(split_tensors[1].allclose(
    ///     &Tensor::<f64>::new(vec![4.0, 5.0, 6.0])
    ///         .reshape(&[1, 1, 3])
    ///         .unwrap()
    /// ));
    /// ```
    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, ErrHandler> {
        Ok(
            _Tensor
                ::hsplit(self.inner.as_ref(), indices)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    /// Splits the tensor into multiple sub-tensors along the vertical axis (first dimension).
    ///
    /// This function divides the tensor along its first dimension (rows) based on the provided `indices`.
    /// It is typically used for 2D tensors (matrices) or higher.
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of indices at which to split the tensor along the vertical axis.
    ///
    /// # Returns
    ///
    /// * A vector of sub-tensors (`Vec<Output>`) split along the vertical axis.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor has fewer than 2 dimensions or if any of the indices are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[2, 3])
    ///     .unwrap();
    /// let split_tensors = tensor.vsplit(&[1]).unwrap();
    /// assert!(split_tensors[0].allclose(
    ///     &Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///         .reshape(&[1, 3])
    ///         .unwrap()
    /// ));
    /// assert!(split_tensors[1].allclose(
    ///     &Tensor::<f64>::new(vec![4.0, 5.0, 6.0])
    ///         .reshape(&[1, 3])
    ///         .unwrap()
    /// ));
    /// ```
    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, ErrHandler> {
        Ok(
            _Tensor
                ::vsplit(self.inner.as_ref(), indices)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    /// Swaps two axes of the tensor, effectively transposing the data along the specified axes.
    ///
    /// This operation exchanges the data between `axis1` and `axis2`, rearranging the tensor without allocating new memory.
    ///
    /// # Arguments
    ///
    /// * `axis1` - The first axis to be swapped.
    /// * `axis2` - The second axis to be swapped.
    ///
    /// # Returns
    ///
    /// * The tensor with the specified axes swapped.
    ///
    /// # Panics
    ///
    /// * This function will panic if either `axis1` or `axis2` are out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]).unwrap();
    /// let transposed_tensor = tensor.swap_axes(0, 1).unwrap();
    /// assert_eq!(transposed_tensor.shape().inner(), &[3, 1]);
    /// ```
    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::swap_axes(self.inner.as_ref(), axis1, axis2)?.into())
    }

    /// Flattens a range of dimensions into a single dimension.
    ///
    /// This operation reduces a subset of dimensions in the tensor, starting from `start` to `end`, into a single dimension.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting axis to begin flattening. Can be `None` to flatten from the first axis.
    /// * `end` - The ending axis for flattening. Can be `None` to flatten up to the last axis.
    ///
    /// # Returns
    ///
    /// * The tensor with the specified range of dimensions flattened.
    ///
    /// # Panics
    ///
    /// * This function will panic if `start` or `end` are out of bounds or if `start` is greater than `end`.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// use tensor_dyn::TensorInfo;
    /// let tensor = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0])
    ///     .reshape(&[2, 2])
    ///     .unwrap();
    /// let flattened_tensor = tensor.flatten(None, None).unwrap();
    /// assert_eq!(flattened_tensor.shape().inner(), &[4]);
    /// ```
    fn flatten<A>(&self, start: A, end: A) -> std::result::Result<Self::Output, ErrHandler> where A: Into<Option<usize>> {
        Ok(_Tensor::flatten(self.inner.as_ref(), start, end)?.into())
    }

    /// Concatenates multiple tensors along a specified axis.
    ///
    /// This operation combines a list of tensors into one, stacking them along the specified `axis`.
    /// All tensors must have the same shape, except for the size along the concatenating axis.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A vector of tensors to concatenate. All tensors must have the same shape, except for the size along the concatenating axis.
    /// * `axis` - The axis along which to concatenate the tensors.
    /// * `keepdims` - A boolean value indicating whether to keep the dimensionality after concatenation.
    ///
    /// # Returns
    ///
    /// * The concatenated tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensors have incompatible shapes along the specified axis.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor1 = Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///    .reshape(&[1, 3])
    ///   .unwrap();
    /// let tensor2 = Tensor::<f64>::new(vec![4.0, 5.0, 6.0])
    ///   .reshape(&[1, 3])
    ///  .unwrap();
    /// let stacked_tensor = Tensor::concat(vec![&tensor1, &tensor2], 0, false).unwrap();
    /// assert!(stacked_tensor.allclose(
    ///   &Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///    .reshape(&[2, 3])
    ///  .unwrap()
    /// ));
    /// ```
    fn concat(tensors: Vec<&Self>, axis: usize, keepdims: bool) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(
            concat(
                tensors
                    .iter()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                axis,
                keepdims
            )?.into()
        )
    }
    /// Stacks multiple tensors vertically (along the first axis).
    ///
    /// This function concatenates tensors by stacking them along the first dimension (rows).
    ///
    /// # Arguments
    ///
    /// * `tensors` - A vector of tensors to stack vertically. All tensors must have the same shape except for the first dimension.
    ///
    /// # Returns
    ///
    /// * The vertically stacked tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensors have incompatible shapes along the first dimension.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor1 = Tensor::<f64>::new(vec![1.0, 2.0, 3.0])
    ///     .reshape(&[1, 3])
    ///     .unwrap();
    /// let tensor2 = Tensor::<f64>::new(vec![4.0, 5.0, 6.0])
    ///     .reshape(&[1, 3])
    ///     .unwrap();
    /// let stacked_tensor = Tensor::vstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(stacked_tensor.allclose(
    ///     &Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///         .reshape(&[2, 3])
    ///         .unwrap()
    /// ));
    /// ```
    fn vstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(
            concat(
                tensors
                    .iter()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                0,
                false
            )?.into()
        )
    }

    /// Stacks multiple tensors horizontally (along the second axis).
    ///
    /// This function concatenates tensors by stacking them along the second dimension (columns).
    ///
    /// # Arguments
    ///
    /// * `tensors` - A vector of tensors to stack horizontally. All tensors must have the same shape except for the second dimension.
    ///
    /// # Returns
    ///
    /// * The horizontally stacked tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensors have incompatible shapes along the second dimension.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor1 = Tensor::<f64>::new(vec![1.0, 2.0]).reshape(&[2, 1]).unwrap();
    /// let tensor2 = Tensor::<f64>::new(vec![3.0, 4.0]).reshape(&[2, 1]).unwrap();
    /// let stacked_tensor = Tensor::hstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(stacked_tensor.allclose(&Tensor::<f64>::new([[1.0, 3.0], [2.0, 4.0]])));
    /// ```
    fn hstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(
            concat(
                tensors
                    .iter()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                1,
                false
            )?.into()
        )
    }

    /// Stacks multiple tensors along the depth axis (third dimension).
    ///
    /// This function concatenates tensors by stacking them along the third dimension (depth).
    ///
    /// # Arguments
    ///
    /// * `tensors` - A vector of tensors to stack along the depth axis. All tensors must have the same shape except for the third dimension.
    ///
    /// # Returns
    ///
    /// * The depth-stacked tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensors have incompatible shapes along the third dimension.
    ///
    /// # Examples
    /// ```
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::ShapeManipulate;
    /// let tensor1 = Tensor::<f64>::new([1.0, 2.0]).reshape(&[2, 1, 1]).unwrap();
    /// let tensor2 = Tensor::<f64>::new([3.0, 4.0]).reshape(&[2, 1, 1]).unwrap();
    /// let stacked_tensor = Tensor::dstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(stacked_tensor.allclose(&Tensor::<f64>::new([[[1.0, 3.0]], [[2.0, 4.0]]])));
    /// ```
    fn dstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(
            concat(
                tensors
                    .iter()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                2,
                false
            )?.into()
        )
    }
}
