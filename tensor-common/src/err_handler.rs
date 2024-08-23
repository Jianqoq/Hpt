use thiserror::Error;

use crate::{shape::Shape, strides::Strides};
#[derive(Debug, Error)]
pub enum ErrHandler {
    #[error("size mismatched: expect {0} but got {1}")] SizeMismatched(i64, i64),
    #[error(
        "matmul shape mismatched: lhs matrix shape is {0:?}, rhs matrix shape is {1:?}, expect rhs matrix shape to be [{2}, any]"
    )] MatmulShapeMismatched([i64; 2], [i64; 2], i64),
    #[error("ndim mismatched: expect {0} but got {1}")] NdimMismatched(usize, usize),
    #[error("axis out of range: tensor ndim is {0} but got index {1} => {2}")] IndexOutOfRange(
        usize,
        i64,
        i64,
    ),
    #[error("Shape mismatched: {0}")] IndexRepeated(String),
    #[error("Shape mismatched: {0}")] ExpandDimError(String),
    #[error("perform reshape error: can't perform inplace reshape to {0} from shape {0} and strides {1}")] IterInplaceReshapeError(Shape, Shape, Strides),
    #[error("broadcast error: {0}")] BroadcastError(String),
    #[error("same axis error: axis should be unique, but got {0} and {1}")] SameAxisError(i64, i64),
}

impl ErrHandler {
    pub fn check_ndim_match(ndim: usize, expect_ndim: usize) -> Result<(), Self> {
        if ndim != expect_ndim {
            return Err(ErrHandler::NdimMismatched(expect_ndim, ndim));
        }
        Ok(())
    }
    pub fn check_same_axis(axis1: i64, axis2: i64) -> Result<(), Self> {
        if axis1 == axis2 {
            return Err(ErrHandler::SameAxisError(axis1, axis2));
        }
        Ok(())
    }
    pub fn check_index_in_range(ndim: usize, index: &mut i64) -> Result<(), Self> {
        let indedx = if *index < 0 { *index + (ndim as i64) } else { *index };
        if indedx < 0 || indedx >= (ndim as i64) {
            return Err(ErrHandler::IndexOutOfRange(ndim, *index, indedx));
        }
        *index = indedx;
        Ok(())
    }
    pub fn check_size_match(size1: i64, size2: i64) -> Result<(), Self> {
        if size1 != size2 {
            return Err(ErrHandler::SizeMismatched(size1, size2));
        }
        Ok(())
    }
}
