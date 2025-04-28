use hpt_common::{
    layout::layout::Layout, shape::shape::Shape, strides::strides::Strides, utils::pointer::Pointer,
};
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    type_promote::{
        FloatOutBinary, FloatOutBinaryPromote, FloatOutUnary, FloatOutUnaryPromote, NormalOut,
        NormalOutPromote, NormalOutUnary,
    },
};
use std::fmt::Debug;
use std::fmt::Display;

/// A trait for getting information of a Tensor
pub trait TensorInfo {
    /// Returns a pointer to the tensor's first data.
    #[track_caller]
    fn ptr<T>(&self) -> Pointer<T>;

    /// Returns the size of the tensor based on the shape
    #[track_caller]
    fn size(&self) -> usize;

    /// Returns the shape of the tensor.
    #[track_caller]
    fn shape(&self) -> &Shape;

    /// Returns the strides of the tensor.
    #[track_caller]
    fn strides(&self) -> &Strides;

    /// Returns the layout of the tensor. Layout contains shape and strides.
    #[track_caller]
    fn layout(&self) -> &Layout;
    /// Returns the root tensor, if any.
    ///
    /// if the tensor is a view, it will return the root tensor. Otherwise, it will return None.
    #[track_caller]
    fn parent<T>(&self) -> Option<Pointer<T>>;

    /// Returns the number of dimensions of the tensor.
    #[track_caller]
    fn ndim(&self) -> usize;

    /// Returns whether the tensor is contiguous in memory. View or transpose tensors are not contiguous.
    #[track_caller]
    fn is_contiguous(&self) -> bool;
}

/// A trait for let the object like a tensor
pub trait TensorLike<T>: Sized {
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

/// Common bounds for primitive types
pub trait CommonBounds
where
    <Self as TypeCommon>::Vec: Send + Sync + Copy,
    Self: Sync
        + Send
        + Clone
        + Copy
        + TypeCommon
        + 'static
        + Display
        + Debug
        + Cast<Self>
        + NormalOut<Self, Output = Self>
        + FloatOutUnary
        + NormalOut<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<Self>
        + NormalOut<
            <Self as FloatOutBinary<Self>>::Output,
            Output = <Self as FloatOutBinary<Self>>::Output,
        >
        + NormalOutUnary
        + FloatOutUnaryPromote
        + FloatOutBinaryPromote
        + NormalOutPromote
        + Cast<f64>,
{
}
impl<T> CommonBounds for T
where
    <Self as TypeCommon>::Vec: Send + Sync + Copy,
    Self: Sync
        + Send
        + Clone
        + Copy
        + TypeCommon
        + 'static
        + Display
        + Debug
        + Cast<Self>
        + NormalOut<Self, Output = Self>
        + FloatOutUnary
        + NormalOut<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<<Self as FloatOutUnary>::Output, Output = <Self as FloatOutUnary>::Output>
        + FloatOutBinary<Self>
        + FloatOutBinary<
            <Self as FloatOutBinary<Self>>::Output,
            Output = <Self as FloatOutBinary<Self>>::Output,
        >
        + NormalOut<
            <Self as FloatOutBinary<Self>>::Output,
            Output = <Self as FloatOutBinary<Self>>::Output,
        >
        + NormalOutUnary
        + FloatOutUnaryPromote
        + FloatOutBinaryPromote
        + NormalOutPromote
        + Cast<f64>,
{
}
