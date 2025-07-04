use crate::{error::base::TensorError, layout::layout::Layout};

/// # Internal Function
/// Processes tensor slicing with given strides and shape, adjusting strides and shape
/// based on the slicing operation and applying an additional scaling factor `alpha`.
///
/// This function performs slicing operations on a tensor's shape and strides according to
/// the provided `index` and scales both the shape and strides by a factor of `alpha`.
///
/// # Arguments
/// - `shape`: A `Vec<i64>` representing the shape of the tensor.
/// - `strides`: A `Vec<i64>` representing the original strides of the tensor.
/// - `index`: A slice of `Slice` enums that specify the slicing operations to apply to each dimension.
/// - `alpha`: A scaling factor of type `i64` that is applied to both the shape and strides.
///
/// # Returns
/// This function returns a `Result` with the following tuple upon success:
/// - `Vec<i64>`: The new shape of the tensor after applying the slicing and scaling.
/// - `Vec<i64>`: The new strides after applying the slicing and scaling.
/// - `i64`: The adjusted pointer offset based on the slicing.
///
/// If the `index` length is out of range for the given `shape`, it returns an error.
///
/// # Errors
/// - Returns an error if the `index` length exceeds the number of dimensions in the tensor shape.
/// - Returns an error if a slicing operation goes out of the bounds of the tensor's shape.
///
/// # Examples
/// ```
/// use hpt_common::slice_process;
/// use hpt_types::Slice;
///
/// let shape = vec![3, 4, 5];
/// let strides = vec![20, 5, 1];
/// let index = vec![Slice::From(1), Slice::Range((0, 3)), Slice::StepByFullRange(2)];
/// let alpha = 1;
/// let result = slice_process(shape, strides, &index, alpha).unwrap();
/// assert_eq!(result, (vec![2, 3, 3], vec![20, 5, 2], 20));
/// ```
#[track_caller]
pub fn slice_process(
    layout: &Layout,
    index: &[(i64, i64, i64)],
    alpha: i64,
) -> std::result::Result<(Vec<i64>, Vec<i64>, i64), TensorError> {
    let mut res_shape: Vec<i64> = layout.shape().to_vec();
    let mut res_strides: Vec<i64> = layout.strides().to_vec();
    res_shape.iter_mut().for_each(|x| {
        *x *= alpha;
    });
    res_strides.iter_mut().for_each(|x| {
        *x *= alpha;
    });
    let mut res_ptr = 0;
    if index.len() > res_shape.len() {
        panic!(
            "index length is greater than the shape length, shape: {:?}, index: {:?}",
            layout.shape(),
            index
        );
    }
    let mut new_indices = Vec::with_capacity(layout.shape().len());
    let ellipsis_pos = index
        .iter()
        .position(|&idx| idx == (0, 0, 0x7FFFFFFFFFFFFFFF));
    if let Some(pos) = ellipsis_pos {
        let missing_dims = layout.shape().len() - (index.len() - 1);
        new_indices.extend_from_slice(&index[0..pos]);
        for _ in 0..missing_dims {
            new_indices.push((0, 0x7FFFFFFFFFFFFFFF, 1));
        }
        new_indices.extend_from_slice(&index[pos + 1..]);
    } else {
        new_indices = index.to_vec();
    }

    for (idx, (start, mut end, step)) in new_indices.into_iter().enumerate() {
        if end == 0x7FFFFFFFFFFFFFFF {
            end = layout.shape()[idx];
        }
        let mut start = if start >= 0 {
            start
        } else {
            start + layout.shape()[idx]
        };
        let mut end = if end >= 0 { end } else { end + layout.shape()[idx] };

        if start >= layout.shape()[idx] {
            start = layout.shape()[idx] - 1;
        }
        if end > layout.shape()[idx] {
            end = layout.shape()[idx];
        }

        let length = if step > 0 {
            (end - start + step - 1) / step
        } else if step < 0 {
            (end - start + step + 1) / step
        } else {
            0
        };
        if length > 0 {
            res_shape[idx] = length * alpha;
            res_ptr += start * res_strides[idx];
            res_strides[idx] *= step;
        } else {
            res_shape[idx] = 0;
        }
    }
    let mut new_shape = Vec::new();
    let mut new_strides = Vec::new();
    for (i, &s) in res_shape.iter().enumerate() {
        if s == 0 {
            continue;
        }
        new_shape.push(s);
        new_strides.push(res_strides[i]);
    }
    Ok((new_shape, new_strides, res_ptr))
}
