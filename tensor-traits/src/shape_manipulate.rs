use tensor_common::{axis::Axis, err_handler::ErrHandler, shape::Shape};

/// A trait for manipulating the shape of a tensor.
pub trait ShapeManipulate
where
    Self: Sized,
{
    /// tensor data type
    type Meta;
    /// the output type
    type Output;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, ErrHandler>;

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
    /// # See Also
    /// - [`permute`]: Rearranges all axes of the tensor according to a given order.
    /// - [`swap_axes`]: Swaps two specified axes in the tensor (an alias for `transpose`).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn t(&self) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mt(&self) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fliplr(&self) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flipud(&self) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tile<S: Into<Axis>>(&self, reps: S) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn trim_zeros(&self, trim: &str) -> std::result::Result<Self::Output, ErrHandler>
    where
        Self::Meta: PartialEq;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn repeat(&self, repeats: usize, axis: i16) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> std::result::Result<Vec<Self::Output>, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self::Output>, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self::Output>, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self::Output>, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flatten<A>(&self, start: A, end: A) -> std::result::Result<Self::Output, ErrHandler>
    where
        A: Into<Option<usize>>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn concat(
        tensors: Vec<&Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn vstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, ErrHandler>;

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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn dstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, ErrHandler>;
}
