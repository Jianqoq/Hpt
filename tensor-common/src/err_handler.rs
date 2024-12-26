use std::panic::Location;

use thiserror::Error;

use crate::{axis::Axis, shape::Shape, strides::Strides};

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

    /// used when the arg reduce error
    #[error("arg reduce error, arg reduce dimension must be 1 but got axis: {0:?}, at {1}")]
    ArgReduceErr(Axis, &'static Location<'static>),

    /// used when the axes is out of bounds
    #[error("axes[{0}][{1}] out of bounds, at {2}")]
    TensorDotAxesOutOfBounds(usize, usize, &'static Location<'static>),

    /// used when the dim is mismatched
    #[error("dim is mismatched, expect lhs_shape[axes[0][{0}]] == rhs_shape[axes[0][{0}]] but got {1} and {2}, at {3}")]
    TensorDotDimMismatched(usize, usize, usize, &'static Location<'static>),

    /// used when the lhs ndim is not compatible with the rhs ndim
    #[error("expect ndim to be {0} but got {1}, at {2}")]
    NdimMismatched(usize, usize, &'static Location<'static>),

    /// used when the ndim is not large enough
    #[error("expect ndim at least {0} but got {1}")]
    NdimNotEnough(usize, usize, &'static Location<'static>),

    /// used when the ndim is too large
    #[error("expect ndim at most {0} but got {1}")]
    NdimExceed(usize, usize, &'static Location<'static>),

    /// used when the tensor is not contiguous
    #[error("tensor is not contiguous, got shape: {0}, strides: {1}, at {2}")]
    ContiguousError(Shape, Strides, &'static Location<'static>),

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

    /// used when the slice index length doesn't match the dimension
    #[error(
        "slice index length doesn't match the dimension, slice index: {0}, dimension: {1}, at {2}"
    )]
    SliceIndexLengthNotMatch(i64, i64, &'static Location<'static>),

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

    /// used when the k is larger than the inner loop size
    #[error("k is larger than the inner loop size, k: {0}, inner loop size: {1}, at {2}")]
    KLargerThanInnerLoopSize(usize, usize, &'static Location<'static>),

    /// used when the environment variable is not set
    #[error("environment variable {0} is not set when calling {1}, at {2}")]
    EnvVarNotSet(&'static str, &'static str, &'static Location<'static>),

    /// used when the cuda kernel compile error
    #[error("cuda kernel compile error, module: {0}, code: {1}, at {2}")]
    CudaKernelCompileError(String, String, &'static Location<'static>),

    #[cfg(feature = "cuda")]
    /// used when the cuda load ptx failed
    #[error("cuda load ptx failed, module: {0}, code: {1}, at {2}. cuda error: {3}")]
    CudaLoadPTXFailed(
        String,
        String,
        &'static Location<'static>,
        cudarc::driver::result::DriverError,
    ),

    /// used when the cuda compile lock failed
    #[error("lock failed in {0}, at {1}")]
    LockFailed(&'static str, &'static Location<'static>),

    /// used when the std::alloc::Layout is not valid
    #[error(
        "std::alloc::Layout is not valid, align: {0}, size: {1}, at {2}. std::LayoutError: {3}"
    )]
    StdMemLayoutError(
        usize,
        usize,
        &'static Location<'static>,
        std::alloc::LayoutError,
    ),

    /// used when the memory allocation failed
    #[error("Failed to allocate {0} memory, for {1} MB, at {2}")]
    MemAllocFailed(&'static str, usize, &'static Location<'static>),

    #[cfg(feature = "cuda")]
    /// used when the cuda rc memory allocation failed
    #[error("cudarc failed to allocate memory, for {0} MB, at {1}. cuda error: {2}")]
    CudaRcMemAllocFailed(
        usize,
        &'static Location<'static>,
        cudarc::driver::result::DriverError,
    ),

    /// used when the reference count overflow
    #[error("reference count overflow for {0}, at {1}")]
    ReferenceCountOverflow(&'static str, &'static Location<'static>),

    #[cfg(feature = "cuda")]
    /// used when the cuda kernel register info is not found
    #[error("cuda kernel register info not found, module: {0}, func: {1}, at {2}")]
    CudaKernelReginfoNotFound(String, String, &'static Location<'static>),

    #[cfg(feature = "cuda")]
    /// used when the cuda kernel meta is not found
    #[error("cuda kernel meta not found for cap: {0}, module: {1}, func: {2}, at {3}")]
    CudaKernelMetaNotFound(usize, String, String, &'static Location<'static>),

    #[cfg(feature = "cuda")]
    /// used when the cuda kernel launching error
    #[error("cuda kernel launching error, module: {0}, func: {1}, at {2}. cuda error: {3}")]
    CudaKernelLaunchingError(
        String,
        String,
        &'static Location<'static>,
        cudarc::driver::result::DriverError,
    ),

    #[cfg(feature = "cuda")]
    /// used when the cuda host to device error
    #[error("cuda data from host to device error, at {0}. cuda error: {1}")]
    CudaHostToDeviceError(
        &'static Location<'static>,
        cudarc::driver::result::DriverError,
    ),

    #[cfg(feature = "cuda")]
    /// used when the cuda create cublas handle error
    #[error("cuda create cublas handle error, at {0}. cuda error: {1}")]
    CudaCreateCublasHandleError(
        &'static Location<'static>,
        cudarc::cublas::result::CublasError,
    ),

    #[cfg(feature = "cuda")]
    /// used when the cuda cublas execute error
    #[error("cuda cublas execute error, at {0}. cuda error: {1}")]
    CudaCublasExecuteError(
        &'static Location<'static>,
        cudarc::cublas::result::CublasError,
    ),

    /// used when the geomspace error
    #[error("geomspace error, start: {0}, end: {1}, they must have the same sign, at {2}")]
    GeomSpaceStartEndError(f64, f64, &'static Location<'static>),

    /// used when the concat dimension is not the same
    #[error("concat dimension is not the same, expect {0} but got {1}, at {2}. Concat requires all except the axis to be the same")]
    ConcatError(usize, usize, &'static Location<'static>),

    /// used when the trim error
    #[error("trim error, trim must be one of 'fb', 'f', 'b', but got {0}, at {1}")]
    TrimError(String, &'static Location<'static>),
}

impl ErrHandler {
    /// panic the error
    pub fn panic(&self) -> ! {
        panic!("{}", self);
    }

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
            return if index < 0 {
                Err(ErrHandler::IndexOutOfRangeCvt(
                    ndim,
                    index,
                    indedx,
                    Location::caller(),
                ))
            } else {
                Err(ErrHandler::IndexOutOfRange(ndim, index, Location::caller()))
            };
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
            return if *index < 0 {
                Err(ErrHandler::IndexOutOfRangeCvt(
                    ndim,
                    *index,
                    indedx,
                    Location::caller(),
                ))
            } else {
                Err(ErrHandler::IndexOutOfRange(
                    ndim,
                    *index,
                    Location::caller(),
                ))
            };
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

    /// function to check if the inplace output is valid
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn check_inplace_out_layout_valid(
        out_shape: &Shape,
        inplace_layout: &crate::layout::Layout,
    ) -> Result<(), Self> {
        if out_shape.size() != inplace_layout.size() {
            return Err(ErrHandler::SizeMismatched(
                out_shape.size(),
                inplace_layout.size(),
                Location::caller(),
            ));
        } else if !inplace_layout.is_contiguous() {
            return Err(ErrHandler::ContiguousError(
                inplace_layout.shape().clone(),
                inplace_layout.strides().clone(),
                Location::caller(),
            ));
        }
        Ok(())
    }
}
