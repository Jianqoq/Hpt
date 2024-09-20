use anyhow::Result;
use tensor_common::{axis::Axis, shape::Shape};

pub trait ShapeManipulate<Output = Self>
where
    Self: Sized,
{
    type Meta;

    /// Squeezes the tensor, removing dimensions of size 1 along the specified axes.
    ///
    /// The `squeeze` function removes dimensions of size 1 from the tensor along the specified axes.
    /// This is useful for simplifying the shape of the tensor without affecting the actual data.
    ///
    /// # Parameters
    ///
    /// - `axes`: The axes along which to remove dimensions of size 1. Can be a single axis or a collection of axes.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified dimensions of size 1 removed.
    ///
    /// # See Also
    ///
    /// - [`unsqueeze`]: Adds dimensions of size 1 to the tensor at the specified axes.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Unsqueezes the tensor, adding dimensions of size 1 along the specified axes.
    ///
    /// The `unsqueeze` function adds dimensions of size 1 to the tensor at the specified axes.
    /// This is useful for expanding the shape of the tensor to match other tensors for broadcasting purposes.
    ///
    /// # Parameters
    ///
    /// - `axes`: The axes along which to add dimensions of size 1. Can be a single axis or a collection of axes.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified dimensions of size 1 added.
    ///
    /// # See Also
    ///
    /// - [`squeeze`]: Removes dimensions of size 1 from the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Reshapes the tensor to a new shape.
    ///
    /// The `reshape` function changes the shape of the tensor to the specified shape, without altering the data.
    /// The total number of elements in the tensor must remain the same in the new shape.
    ///
    /// # Parameters
    ///
    /// - `shape`: The new shape for the tensor. It can be a fixed-size array or a vector representing the target shape.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified shape.
    ///
    /// # See Also
    ///
    /// - [`transpose`]: Swaps two specified axes in the tensor.
    /// - [`expand`]: Expands the tensor to a larger shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Output>;

    /// Transposes two axes of the tensor.
    ///
    /// The `transpose` function swaps two specified axes in the tensor, effectively transposing those dimensions.
    /// This is useful for changing the order of dimensions in multi-dimensional tensors.
    ///
    /// # Parameters
    ///
    /// - `axis1`: The first axis to swap.
    /// - `axis2`: The second axis to swap.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified axes swapped.
    ///
    /// # See Also
    ///
    /// - [`permute`]: Rearranges all axes of the tensor according to a given order.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn transpose(&self, axis1: i64, axis2: i64) -> Result<Output>;

    /// Permutes the axes of the tensor according to a specified order.
    ///
    /// The `permute` function rearranges the axes of the tensor according to the specified order.
    /// This is useful for changing the shape of the tensor to match other tensors or for certain operations.
    ///
    /// # Parameters
    ///
    /// - `axes`: A list of axes representing the new order. It can be a fixed-size array or a vector representing the target axes order.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified axes order.
    ///
    /// # See Also
    ///
    /// - [`permute_inv`]: Reverses the effect of the permute function by rearranging the axes back to their original order.
    /// - [`transpose`]: Swaps two specified axes in the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Permutes the axes of the tensor back to their original order.
    ///
    /// The `permute_inv` function reverses the effect of the `permute` function, restoring the original order of axes.
    ///
    /// # Parameters
    ///
    /// - `axes`: A list of axes representing the new order. It can be a fixed-size array or a vector representing the target axes order.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the original axes order restored.
    ///
    /// # See Also
    ///
    /// - [`permute`]: Rearranges all axes of the tensor according to a given order.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Expands the tensor to the specified shape.
    ///
    /// The `expand` function expands the tensor to a larger shape by repeating its elements along specified dimensions,
    /// without copying the underlying data.
    ///
    /// # Parameters
    ///
    /// - `shape`: The target shape for the expanded tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the specified expanded shape.
    ///
    /// # See Also
    ///
    /// - [`reshape`]: Reshapes the tensor to a new shape without altering the data.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Output>;

    /// Transposes the last two dimensions of the tensor.
    ///
    /// The `t` function swaps the last two dimensions of the tensor, commonly used for 2D tensors like matrices.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the last two dimensions swapped.
    ///
    /// # See Also
    ///
    /// - [`mt`]: A shorthand for matrix transpose that also works for batch dimensions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn t(&self) -> Result<Output>;

    /// Reverse the order of the dimensions of the tensor.
    ///
    /// The `mt` function reverses the order of all the dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the last two dimensions swapped.
    ///
    /// # See Also
    ///
    /// - [`t`]: Transposes the last two dimensions of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mt(&self) -> Result<Output>;

    /// Flips the tensor along the specified axes.
    ///
    /// The `flip` function reverses the elements of the tensor along the specified axes, effectively flipping the tensor.
    ///
    /// # Parameters
    ///
    /// - `axes`: The axes along which to flip the tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the elements flipped along the specified axes.
    ///
    /// # See Also
    ///
    /// - [`fliplr`]: Flips the tensor along the horizontal axis.
    /// - [`flipud`]: Flips the tensor along the vertical axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Flips the tensor along the horizontal axis (left to right).
    ///
    /// The `fliplr` function reverses the elements of the tensor along the horizontal axis, flipping the tensor from left to right.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the elements flipped horizontally.
    ///
    /// # See Also
    ///
    /// - [`flipud`]: Flips the tensor along the vertical axis (up to down).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fliplr(&self) -> Result<Output>;

    /// Flips the tensor along the vertical axis (up to down).
    ///
    /// The `flipud` function reverses the elements of the tensor along the vertical axis, flipping the tensor from top to bottom.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the elements flipped vertically.
    ///
    /// # See Also
    ///
    /// - [`fliplr`]: Flips the tensor along the horizontal axis (left to right).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flipud(&self) -> Result<Output>;

    /// Tiles the tensor by repeating it along specified axes.
    ///
    /// The `tile` function repeats the tensor along the specified axes according to the number of repetitions provided.
    ///
    /// # Parameters
    ///
    /// - `reps`: The number of repetitions along each axis. Can be a single number or a collection of values representing repetitions along multiple axes.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the tiled elements.
    ///
    /// # See Also
    ///
    /// - [`repeat`]: Repeats the elements of the tensor along a single axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tile<S: Into<Axis>>(&self, reps: S) -> Result<Output>;
    /// Trims leading and/or trailing zeros from the tensor.
    ///
    /// The `trim_zeros` function removes zeros from the tensor, either from the start, the end, or both, depending on the `trim` argument.
    ///
    /// # Parameters
    ///
    /// - `trim`: A string indicating which zeros to trim. Can be `"start"`, `"end"`, or `"both"`.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the leading and/or trailing zeros removed.
    ///
    /// # Notes
    ///
    /// - **Trimming Zeros**: Zeros are removed according to the specified option. Non-zero elements remain unchanged.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn trim_zeros(&self, trim: &str) -> Result<Output>
    where
        Self::Meta: PartialEq;

    /// Repeats elements of the tensor along a specified axis.
    ///
    /// The `repeat` function repeats the elements of the tensor along the specified axis a given number of times.
    ///
    /// # Parameters
    ///
    /// - `repeats`: The number of times to repeat each element.
    /// - `axis`: The axis along which to repeat the elements.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor with the elements repeated along the specified axis.
    ///
    /// # See Also
    ///
    /// - [`tile`]: Tiles the entire tensor by repeating it along multiple axes.

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn repeat(&self, repeats: usize, axis: i16) -> Result<Output>;

    /// Splits the tensor into multiple sub-tensors along a specified axis.
    ///
    /// The `split` function splits the tensor into multiple sub-tensors based on the specified indices or sections.
    ///
    /// # Parameters
    ///
    /// - `indices_or_sections`: Either a list of indices or a number of sections to split the tensor into.
    /// - `axis`: The axis along which to split the tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Vec<Output>>`: A list of sub-tensors resulting from the split.
    ///
    /// # See Also
    ///
    /// - [`dsplit`]: Splits the tensor along the depth axis.
    /// - [`hsplit`]: Splits the tensor along the horizontal axis.
    /// - [`vsplit`]: Splits the tensor along the vertical axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Output>>;

    /// Splits the tensor along the depth axis (axis 0).
    ///
    /// The `dsplit` function splits the tensor into multiple sub-tensors along the depth axis (axis 0) at the specified indices.
    ///
    /// # Parameters
    ///
    /// - `indices`: The indices at which to split the tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Vec<Output>>`: A list of sub-tensors resulting from the split.
    ///
    /// # See Also
    ///
    /// - [`split`]: Splits the tensor along a specified axis.
    /// - [`hsplit`]: Splits the tensor along the horizontal axis.
    /// - [`vsplit`]: Splits the tensor along the vertical axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Output>>;

    /// Splits the tensor along the horizontal axis (axis 1).
    ///
    /// The `hsplit` function splits the tensor into multiple sub-tensors along the horizontal axis (axis 1) at the specified indices.
    ///
    /// # Parameters
    ///
    /// - `indices`: The indices at which to split the tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Vec<Output>>`: A list of sub-tensors resulting from the split.
    ///
    /// # See Also
    ///
    /// - [`split`]: Splits the tensor along a specified axis.
    /// - [`dsplit`]: Splits the tensor along the depth axis.
    /// - [`vsplit`]: Splits the tensor along the vertical axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Output>>;

    /// Splits the tensor along the vertical axis (axis 2).
    ///
    /// The `vsplit` function splits the tensor into multiple sub-tensors along the vertical axis (axis 2) at the specified indices.
    ///
    /// # Parameters
    ///
    /// - `indices`: The indices at which to split the tensor.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Vec<Output>>`: A list of sub-tensors resulting from the split.
    ///
    /// # See Also
    ///
    /// - [`split`]: Splits the tensor along a specified axis.
    /// - [`dsplit`]: Splits the tensor along the depth axis.
    /// - [`hsplit`]: Splits the tensor along the horizontal axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Output>>;

    /// alias of `transpose`
    ///
    /// # See Also
    ///
    /// - [`transpose`]: Swaps two specified axes in the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn swap_axes(&self, axis1: i64, axis2: i64) -> Result<Output>;

    /// Flattens the tensor to a 1D array or a sub-range of the tensor's dimensions.
    ///
    /// The `flatten` function flattens the tensor either completely or partially, depending on the specified start and end dimensions.
    ///
    /// # Parameters
    ///
    /// - `start`: The starting dimension for flattening.
    /// - `end`: The ending dimension for flattening.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Output>`: A new tensor that is flattened according to the specified range.
    ///
    /// # See Also
    ///
    /// - [`reshape`]: Reshapes the tensor to a new shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flatten<A>(&self, start: A, end: A) -> Result<Output>
    where
        A: Into<Option<usize>>;

    /// Stacks a sequence of tensors along a specified axis.
    ///
    /// Given a list of tensors, this function concatenates them along the specified axis.
    /// All tensors must have the same shape, except in the dimension corresponding to `axis`.
    ///
    /// # Arguments
    /// - `tensors`: A vector of tensor references to be stacked.
    /// - `axis`: The axis along which the tensors will be stacked.
    /// - `keepdims`: A boolean indicating whether to keep the dimension of the axis or not.
    ///
    /// # Returns
    /// A `Result` containing the stacked tensor or an error if the operation fails.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn concat(tensors: Vec<&Self>, axis: usize, keepdims: bool) -> Result<Output>;

    /// Vertically stacks a sequence of tensors.
    ///
    /// This is a convenience method for stacking tensors along the first axis (axis=0).
    /// All tensors must have the same number of dimensions and the same shape,
    /// except for the first axis.
    ///
    /// # Arguments
    /// - `tensors`: A vector of tensor references to be vertically stacked.
    ///
    /// # Returns
    /// A `Result` containing the vertically stacked tensor or an error if the operation fails.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn vstack(tensors: Vec<&Self>) -> Result<Output>;

    /// Horizontally stacks a sequence of tensors.
    ///
    /// This function concatenates tensors along the second axis (axis=1).
    /// It automatically reshapes tensors with fewer dimensions to have an additional axis.
    /// For 1-dimensional tensors, they are reshaped to 2D before stacking.
    /// Scalars are reshaped to 1x1 tensors.
    ///
    /// # Arguments
    /// - `tensors`: A vector of references to the tensors to be horizontally stacked.
    ///
    /// # Returns
    /// A `Result` containing the horizontally stacked tensor or an error if the operation fails.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hstack(tensors: Vec<&Self>) -> Result<Output>;

    /// Depth-stacks a sequence of tensors.
    ///
    /// This function concatenates tensors along the third axis (axis=2).
    /// It automatically reshapes tensors with fewer dimensions to match the required number of dimensions.
    /// For 1-dimensional tensors, they are reshaped to 1xNx1 before stacking.
    /// For 2-dimensional tensors, they are reshaped to NxMx1.
    /// Scalars are reshaped to 1x1x1 tensors.
    ///
    /// # Arguments
    /// - `tensors`: A vector of references to the tensors to be depth-stacked.
    ///
    /// # Returns
    /// A `Result` containing the depth-stacked tensor or an error if the operation fails.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn dstack(tensors: Vec<&Self>) -> Result<Output>;
}
