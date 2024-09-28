use std::panic::Location;

use thiserror::Error;

use crate::{shape::Shape, strides::Strides};

/// Error handler for the library
///
/// it is used to handle the errors that might occur during the operations
#[derive(Debug, Error)]
pub enum ErrHandler {
    /// used when the size of the tensor is not as expected
    #[error("expect size {0} but got size {1}, at {2}")]
    SizeMismatched(i64, i64, &'static Location<'static>),

    /// used when the lhs matrix shape is not compatible with the rhs matrix shape
    #[error(
        "lhs matrix shape is {0:?}, rhs matrix shape is {1:?}, expect rhs matrix shape to be [{2}, any], at {3}"
    )]
    MatmulShapeMismatched([i64; 2], [i64; 2], i64, &'static Location<'static>),

    /// used when the lhs ndim is not compatible with the rhs ndim
    #[error("expect ndim to be {0} but got {1}, at {2}")]
    NdimMismatched(usize, usize, &'static Location<'static>),

    /// used when the ndim is not large enough
    #[error("expect ndim at least {0} but got {1}")]
    NdimNotEnough(usize, usize, &'static Location<'static>),

    /// used when the ndim is too large
    #[error("expect ndim at most {0} but got {1}")]
    NdimExceed(usize, usize, &'static Location<'static>),

    /// used when the axis is out of range
    #[error("tensor ndim is {0} but got index `{1}`, at {2}")]
    IndexOutOfRange(usize, i64, &'static Location<'static>),

    /// used when the axis is out of range, this is used for out of range when converting the negative axis to positive axis
    #[error("tensor ndim is {0} but got converted index from `{1}` to `{2}`, at {3}")]
    IndexOutOfRangeCvt(usize, i64, i64, &'static Location<'static>),

    /// used when the axis provided is not unique, for example, sum([1, 1]) is not allowed
    #[error("Shape mismatched: {0}")]
    IndexRepeated(String),

    /// used when the shape is not compatible with the strides
    #[error("Shape mismatched: {0}")]
    ExpandDimError(String),

    /// used when trying to reshape the tensor iterator is not possible
    #[error("can't perform inplace reshape to from {0} to {1} with strides {2}, at {3}")]
    IterInplaceReshapeError(Shape, Shape, Strides, &'static Location<'static>),

    /// used when the lhs shape is not possible to broadcast to the rhs shape
    #[error("can't broacast lhs: {0} with rhs: {1}, expect lhs_shape[{2}] to be 1, at {3}")]
    BroadcastError(Shape, Shape, usize, &'static Location<'static>),

    /// used when the axis is not unique
    #[error("axis should be unique, but got {0} and {1}, at {2}")]
    SameAxisError(i64, i64, &'static Location<'static>),

    /// used when the reshape is not possible
    #[error("can't reshape from {0} with size {2} to {1} with size {3}, at {4}")]
    ReshapeError(Shape, Shape, usize, usize, &'static Location<'static>),

    /// used when the transpose is not possible
    #[error("can't transpose {0}, ndim is expected to >= 2 but got {1}, at {2}")]
    TransposeError(Shape, usize, &'static Location<'static>),

    /// used when the slice index is out of range
    #[error("slice index out of range for {0} (arg: {1}), it should < {2}, At {3}")]
    SliceIndexOutOfRange(i64, i64, i64, &'static Location<'static>),

    /// used when the dimension to squeeze is not 1
    #[error(
        "cannot select an axis to squeeze out which has size != 1, found error for index {0} in {1}, at {2}"
    )]
    SqueezeError(usize, Shape, &'static Location<'static>),

    /// used when the dimension is less than 0
    #[error("invalide input shape, result dim can't less than 0, got {0}, at {1}")]
    InvalidInputShape(i64, &'static Location<'static>),

    /// currently only used for conv, max_pool, avg_pool, etc.
    #[error(
        "internal error: invalid cache param, {0} must be less than {1} and multiple of {2} or equal to 1, but got {3}, at {4}"
    )]
    InvalidCacheParam(&'static str, i64, i64, i64, &'static Location<'static>),

    /// used when the conv2d input shape is not correct
    #[error("invalid input shape, expect shape to be [batch, height, width, channel], but got ndim: {0}, at {1}")]
    Conv2dImgShapeInCorrect(usize, &'static Location<'static>),

    /// used when the out pass to the out method is not valid
    #[error("out size is invalid, expect out to be {0} bits but got {1} bits, at {2}")]
    InvalidOutSize(usize, usize, &'static Location<'static>),
}

impl ErrHandler {
    /// function to check if the ndim is same as expected ndim
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn check_ndim_match(ndim: usize, expect_ndim: usize) -> Result<(), Self> {
        if ndim != expect_ndim {
            return Err(ErrHandler::NdimMismatched(
                expect_ndim,
                ndim,
                Location::caller(),
            ));
        }
        Ok(())
    }

    /// function to check if two axis is not the same, if they are the same, it will return an error
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn check_same_axis(axis1: i64, axis2: i64) -> Result<(), Self> {
        if axis1 == axis2 {
            return Err(ErrHandler::SameAxisError(axis1, axis2, Location::caller()));
        }
        Ok(())
    }

    /// function to check if the index provided is in the range of the ndim, if not, it will return an error
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn check_index_in_range(ndim: usize, index: i64) -> Result<(), Self> {
        let indedx = if index < 0 {
            index + (ndim as i64)
        } else {
            index
        };
        if indedx < 0 || indedx >= (ndim as i64) {
            if index < 0 {
                return Err(ErrHandler::IndexOutOfRangeCvt(
                    ndim,
                    index,
                    indedx,
                    Location::caller(),
                ));
            } else {
                return Err(ErrHandler::IndexOutOfRange(ndim, index, Location::caller()));
            }
        }
        Ok(())
    }

    /// function to check if the index provided is in the range of the ndim, if not, it will return an error
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn check_index_in_range_mut(ndim: usize, index: &mut i64) -> Result<(), Self> {
        let indedx = if *index < 0 {
            *index + (ndim as i64)
        } else {
            *index
        };
        if indedx < 0 || indedx >= (ndim as i64) {
            if *index < 0 {
                return Err(ErrHandler::IndexOutOfRangeCvt(
                    ndim,
                    *index,
                    indedx,
                    Location::caller(),
                ));
            } else {
                return Err(ErrHandler::IndexOutOfRange(
                    ndim,
                    *index,
                    Location::caller(),
                ));
            }
        }
        *index = indedx;
        Ok(())
    }

    /// function to check if the size of the tensor is the same as expected size
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn check_size_match(size1: i64, size2: i64) -> Result<(), Self> {
        if size1 != size2 {
            return Err(ErrHandler::SizeMismatched(size1, size2, Location::caller()));
        }
        Ok(())
    }
}
