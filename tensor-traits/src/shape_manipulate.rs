use anyhow::Result;
use tensor_common::{ axis::Axis, shape::Shape };

pub trait ShapeManipulate<Output = Self> where Self: Sized {
    type Meta;

    /// remove a sequence of dimensions from the shape of a tensor
    ///
    /// # Arguments
    /// `axes` - the axes to be removed
    ///
    /// # Returns
    /// a tensor with the specified axes removed
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.squeeze(0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// yeild `1` dimension to the shape of a tensor in the specific axes
    ///
    /// # Arguments
    /// `axes` - the axes to be added
    ///
    /// # Returns
    /// a tensor with the specified axes added
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.unsqueeze(0).unwrap();
    /// assert_eq!(b.shape(), &[1, 100]);
    /// ```
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Gives a new shape to an array without changing its data.
    ///
    /// # Arguments
    /// `shape` - the new shape
    ///
    /// # Returns
    /// a tensor with the specified shape
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.reshape([10, 10]).unwrap();
    /// assert_eq!(b.shape(), &[10, 10]);
    /// ```
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Output>;

    /// Returns an array with axes swapped.
    /// This method always return a view of the original Tensor
    ///
    /// # Arguments
    /// `axis1` - axis 1 to be swapped
    /// `axis2` - axis 2 to be swapped
    ///
    /// # Returns
    /// a tensor with the specified axes swapped
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.transpose(0, 1).unwrap();
    /// assert_eq!(b.shape(), &[100, 1]);
    /// ```
    fn transpose(&self, axis1: i64, axis2: i64) -> Result<Output>;

    /// Permutes the dimensions of an array.
    /// This method always return a view of the original Tensor
    ///
    /// # Arguments
    /// `axes` - the new order of the axes
    ///
    /// # Returns
    /// a tensor with the specified axes permuted
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.permute([1, 0]).unwrap();
    /// assert_eq!(b.shape(), &[100, 1]);
    /// ```
    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    /// Expand the tensor to the specified shape
    /// only the dimensions that are 1 can be expanded
    ///
    /// # Arguments
    /// `shape` - the new shape
    ///
    /// # Returns
    /// a tensor with the specified shape
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.expand([10, 10]).unwrap();
    /// assert_eq!(b.shape(), &[10, 10]);
    /// ```
    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Output>;

    /// tranpose a Tensor with dimension `>= 2`
    ///
    /// for dimension larger than `2`, the last two dimensions are swapped
    ///
    /// # Returns
    /// a Tensor with the specified axes swapped
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.t().unwrap();
    /// assert_eq!(b.shape(), &[1, 100]);
    /// ```
    fn t(&self) -> Result<Output>;

    /// flip the axes of a tensor
    ///
    /// # Returns
    /// a tensor with all the axes flipped
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.flip(Axis(0)).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    fn mt(&self) -> Result<Output>;

    /// flip the elements of a tensor along a sequence of axes
    ///
    /// # Arguments
    /// `axes` - the axes to be flipped
    ///
    /// # Returns
    /// a tensor with the specified axes flipped
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.flip(Axis(0)).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Output>;

    fn fliplr(&self) -> Result<Output>;
    fn flipud(&self) -> Result<Output>;

    /// Construct a tensor by repeating A the number of times given by reps.
    ///
    /// # Arguments
    /// `reps` - the number of repetitions for each axis
    ///
    /// # Returns
    /// a tensor with the specified axes repeated
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3]).unwrap();
    /// let b = a.tile(3).unwrap();
    /// assert_eq!(b.shape(), &[3, 3]);
    /// ```
    fn tile<S: Into<Axis>>(&self, reps: S) -> Result<Output>;

    /// Trim the leading and/or trailing zeros from a 1-D tensor.
    ///
    /// # Arguments
    /// `trim` - the side(s) of the tensor to be trimmed
    ///
    /// # Returns
    /// a tensor with the specified axes trimmed
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[0, 0, 0, 1, 2, 3, 0, 0, 0]).unwrap();
    /// let b = a.trim_zeros("fb").unwrap();
    /// assert_eq!(b.shape(), &[1, 2, 3]);
    /// ```
    fn trim_zeros(&self, trim: &str) -> Result<Output> where Self::Meta: PartialEq;

    /// Repeat elements along a given axis of a tensor.
    ///
    /// # Arguments
    /// `repeats` - the number of repetitions for each axis
    /// `axis` - the axis to be repeated
    ///
    /// # Returns
    /// a tensor with the specified axes repeated
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3]).unwrap();
    /// let b = a.repeat(3, 0).unwrap();
    /// assert_eq!(b.shape(), &[9]);
    /// ```
    fn repeat(&self, repeats: usize, axis: i16) -> Result<Output>;

    /// Split a tensor into multiple sub-tensors.
    ///
    /// # Arguments
    /// `indices_or_sections` - the indices or sections to be split
    /// `axis` - the axis to be split
    ///
    /// # Returns
    /// a vector of tensors with the specified axes split
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3, 4, 5, 6]).unwrap();
    /// let b = a.split(&[2, 4], 0).unwrap();
    /// assert_eq!(b[0].shape(), &[2]);
    /// assert_eq!(b[1].shape(), &[2]);
    /// assert_eq!(b[2].shape(), &[2]);
    /// ```
    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Output>>;

    /// Split a tensor into multiple sub-tensors along the first axis.
    ///
    /// # Arguments
    /// `indices` - the indices to be split
    ///
    /// # Returns
    /// a vector of tensors with the specified axes split
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3, 4, 5, 6]).unwrap();
    /// let b = a.dsplit(&[2, 4]).unwrap();
    /// assert_eq!(b[0].shape(), &[2, 3]);
    /// assert_eq!(b[1].shape(), &[2, 3]);
    /// assert_eq!(b[2].shape(), &[2, 3]);
    /// ```
    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Output>>;

    /// Split a tensor into multiple sub-tensors along the last axis.
    ///
    /// # Arguments
    /// `indices` - the indices to be split
    ///
    /// # Returns
    /// a vector of tensors with the specified axes split
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3, 4, 5, 6]).unwrap();
    /// let b = a.hsplit(&[2, 4]).unwrap();
    /// assert_eq!(b[0].shape(), &[3, 2]);
    /// assert_eq!(b[1].shape(), &[3, 2]);
    /// assert_eq!(b[2].shape(), &[3, 2]);
    /// ```
    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Output>>;

    /// Split a tensor into multiple sub-tensors along the second axis.
    ///
    /// # Arguments
    /// `indices` - the indices to be split
    ///
    /// # Returns
    /// a vector of tensors with the specified axes split
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3, 4, 5, 6]).unwrap();
    /// let b = a.vsplit(&[2, 4]).unwrap();
    /// assert_eq!(b[0].shape(), &[3, 2]);
    /// assert_eq!(b[1].shape(), &[3, 2]);
    /// assert_eq!(b[2].shape(), &[3, 2]);
    /// ```
    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Output>>;

    /// Swap the axes of a tensor.
    ///
    /// # Arguments
    /// `axis1` - axis 1 to be swapped
    /// `axis2` - axis 2 to be swapped
    ///
    /// # Returns
    /// a tensor with the specified axes swapped
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::from_array(&[1, 2, 3, 4, 5, 6]).unwrap();
    /// let b = a.swap_axes(0, 1).unwrap();
    /// assert_eq!(b.shape(), &[3, 2]);
    /// ```
    fn swap_axes(&self, axis1: i64, axis2: i64) -> Result<Output>;

    fn flatten<A>(&self, axis: A) -> Result<Output> where A: Into<Option<usize>>;
}
