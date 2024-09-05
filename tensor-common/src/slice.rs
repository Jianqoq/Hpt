use std::panic::Location;

use anyhow::Result;

use crate::err_handler::ErrHandler;

/// Slice enum to hold the slice information
///
/// it stores the slice information the user wants to perform operations on
///
/// it is not being used directly by the user, but is used by the library internally
///
/// # Variants
///
/// - `From(i64)` - load the element at the index
///
/// - `Full` - load all the elements along the corresponding dimension
///
/// - `RangeFrom(i64)` - load from the first element to the end along the corresponding dimension
///
/// - `RangeTo(i64)` - load from the beginning to specified index along the corresponding dimension
///
/// - `Range((i64, i64))` - load from the start index to the end index along the corresponding dimension
///
/// - `StepByRangeFrom((i64, i64))` - load from the start index to the end index with step along the corresponding dimension
///
/// - `StepByFullRange(i64)` - load all the elements with step along the corresponding dimension
///
/// - `StepByRangeFromTo((i64, i64, i64))` - load from the start index to the end index with step along the corresponding dimension
///
/// - `StepByRangeTo((i64, i64))` - load from the start index to the end index with step along the corresponding dimension
#[derive(Debug, Clone)]
pub enum Slice {
    From(i64),
    Full,
    RangeFrom(i64),
    RangeTo(i64),
    Range((i64, i64)),
    StepByRangeFrom((i64, i64)),
    StepByFullRange(i64),
    StepByRangeFromTo((i64, i64, i64)),
    StepByRangeTo((i64, i64)),
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub fn slice_process(
    shape: Vec<i64>,
    strides: Vec<i64>,
    index: &[Slice],
    alpha: i64
) -> Result<(Vec<i64>, Vec<i64>, i64)> {
    let mut res_shape: Vec<i64> = shape.clone();
    let mut res_strides: Vec<i64> = strides.clone();
    res_shape.iter_mut().for_each(|x| {
        *x *= alpha;
    });
    res_strides.iter_mut().for_each(|x| {
        *x *= alpha;
    });
    let mut res_ptr = 0;
    if index.len() > res_shape.len() {
        let message = "slice input out of range".to_string();
        return Err(anyhow::Error::msg(message));
    }
    for (idx, slice) in index.iter().enumerate() {
        match slice {
            Slice::From(mut __index) => {
                let mut index;
                if __index >= 0 {
                    index = __index;
                } else {
                    index = __index + shape[idx];
                }
                index *= alpha;
                if index >= shape[idx] {
                    return Err(
                        ErrHandler::SliceIndexOutOfRange(
                            index,
                            idx as i64,
                            shape[idx],
                            Location::caller()
                        ).into()
                    );
                }
                res_shape[idx] = alpha;
                res_ptr += res_strides[idx] * index;
            }
            // tested
            Slice::RangeFrom(mut __index) => {
                let index = if __index >= 0 { __index } else { __index + shape[idx] };
                let length = (shape[idx] - index) * alpha;
                res_shape[idx] = if length > 0 { length } else { 0 };
                res_ptr += res_strides[idx] * index;
            }
            Slice::RangeTo(r) => {
                let range_to = if *r >= 0 { ..*r } else { ..*r + shape[idx] };
                let mut end = range_to.end;
                end *= alpha;
                if range_to.end > res_shape[idx] {
                    end = res_shape[idx];
                }
                res_shape[idx] = end;
            }
            // tested
            Slice::Range((start, end)) => {
                let range;
                if *start >= 0 {
                    if *end >= 0 {
                        range = *start..*end;
                    } else {
                        range = *start..*end + shape[idx];
                    }
                } else if *end >= 0 {
                    range = *start + shape[idx]..*end;
                } else {
                    range = start + shape[idx]..*end + shape[idx];
                }
                let mut start = range.start;
                start *= alpha;
                let mut end = range.end;
                end *= alpha;
                if start >= res_shape[idx] {
                    start = res_shape[idx];
                }
                if end >= res_shape[idx] {
                    end = res_shape[idx];
                }
                if start > end {
                    res_shape[idx] = 0;
                } else {
                    res_shape[idx] = end - start;
                    res_ptr += strides[idx] * start;
                }
            }
            // tested
            Slice::StepByRangeFromTo((start, end, step)) => {
                let mut start = if *start >= 0 { *start } else { *start + shape[idx] };
                let mut end = if *end >= 0 { *end } else { *end + shape[idx] };
                if start >= shape[idx] {
                    start = shape[idx] - 1;
                }
                if end >= shape[idx] {
                    end = shape[idx] - 1;
                }
                let length;
                if start <= end && *step > 0 {
                    length = (end - 1 - start + step) / step;
                } else if start >= end && *step < 0 {
                    length = (end + 1 - start + step) / step;
                } else {
                    length = 0;
                }
                if length == 1 {
                    res_shape[idx] = alpha;
                    res_ptr += res_strides[idx] * start;
                } else if length >= 0 {
                    res_shape[idx] = length * alpha;
                    res_ptr += start * res_strides[idx];
                    res_strides[idx] *= *step;
                } else {
                    res_shape[idx] = 0;
                }
            }
            // tested
            Slice::StepByRangeFrom((start, step)) => {
                let mut start = if *start >= 0 { *start } else { *start + shape[idx] };
                let end = if *step > 0 { shape[idx] } else { 0 };
                if start >= shape[idx] {
                    start = shape[idx] - 1;
                }
                let length;
                if start <= end && *step > 0 {
                    length = (end - 1 - start + step) / step;
                } else if start >= end && *step < 0 {
                    length = (end - start + step) / step;
                } else {
                    length = 0;
                }
                if length == 1 {
                    res_shape[idx] = alpha;
                    res_ptr += res_strides[idx] * start;
                } else if length >= 0 {
                    res_shape[idx] = length * alpha;
                    res_ptr += start * res_strides[idx];
                    res_strides[idx] *= *step;
                } else {
                    res_shape[idx] = 0;
                }
            }
            // tested
            Slice::StepByFullRange(step) => {
                let start = if *step > 0 { 0 } else { shape[idx] - 1 };
                let end = if *step > 0 { shape[idx] - 1 } else { 0 };
                let length = if (start <= end && *step > 0) || (start >= end && *step < 0) {
                    (end - start + step) / step
                } else {
                    0
                };
                if length == 1 {
                    res_shape[idx] = alpha;
                    res_ptr += res_strides[idx] * start;
                } else if length >= 0 {
                    res_shape[idx] = length * alpha;
                    res_ptr += start * res_strides[idx];
                    res_strides[idx] *= *step;
                } else {
                    res_shape[idx] = 0;
                }
            }
            _ => {}
        }
    }
    Ok((res_shape, res_strides, res_ptr))
}

/// slice operation for tensor
/// slicing uses the same syntax as numpy
///
/// `[:::]` and `[:]` and `[::]`: load all the elements along the corresponding dimension
///
/// `[1:]`: load from the first element to the end along the corresponding dimension
///
/// `[:10]`: load from the beginning to index 9 along the corresponding dimension
///
/// `[1:10]`: load from index 1 to index 9 along the corresponding dimension
///
/// `[1:10:2]`: load from index 1 to index 9 with step 2 along the corresponding dimension
///
/// `[1:10:2, 2:10:3]`: load from index 1 to index 9 with step 2 for the first dimension, and load from index 2 to index 9 with step 3 for the second dimension
///
/// `[::2]`: load all the elements with step 2 along the corresponding dimension
/// Example:
/// ```
/// use tensor_core::prelude::*;
/// let a = Tensor::<f32>::rand([128, 128, 128])?;
/// let res = slice!(a[::2]); // load all the elements with step 2 along the first dimension
/// let res = slice!(a[1:10:2, 2:10:3]); // load from index 1 to index 9 with step 2 for the first dimension, and load from index 2 to index 9 with step 3 for the second dimension
/// ```
#[macro_export]
macro_rules! slice {
    (
        $tensor:ident [$($indexes:tt)*]
    ) => {
        $tensor.slice(match_selection!($($indexes)*))
    };
}
