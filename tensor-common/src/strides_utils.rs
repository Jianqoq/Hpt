use tensor_types::{convertion::Convertor, dtype::TypeCommon, into_scalar::IntoScalar};

use crate::strides::Strides;

/// # Internal Function
/// Preprocesses strides based on the given shape.
///
/// This function adjusts the strides of a tensor according to its shape.
/// Strides corresponding to dimensions with size 1 are set to 0.
///
/// # Arguments
/// - `shape`: A reference to the shape of the tensor.
/// - `stride`: A reference to the original strides of the tensor.
///
/// # Returns
/// A `Vec<i64>` representing the preprocessed strides.
///
/// # Examples
/// ```
/// use tensor_common::preprocess_strides;
/// let shape = vec![1, 2, 3];
/// let strides = vec![3, 2, 1];
/// let preprocessed_strides = preprocess_strides(&shape, &strides);
/// ```
pub fn preprocess_strides<
    A: Convertor + Copy,
    B: Convertor + IntoScalar<C> + Copy,
    C: TypeCommon + Copy,
>(
    shape: &[A],
    stride: &[B],
) -> Vec<C> {
    let mut strides = vec![C::ZERO; shape.len()];
    let start = shape.len() - stride.len();

    for i in 0..stride.len() {
        if shape[start + i].to_i64() != 1i64 {
            strides[start + i] = stride[i].into_scalar();
        }
    }
    strides
}

/// # Internal Function
/// Converts a shape to strides.
///
/// This function calculates the strides of a tensor based on its shape,
/// assuming a contiguous memory layout.
///
/// # Arguments
/// - `shape`: A reference to the shape of the tensor.
///
/// # Returns
/// A `Vec<i64>` representing the strides calculated from the shape.
///
/// # Examples
/// ```
/// use tensor_common::shape_to_strides;
/// let shape = vec![2, 3, 4];
/// let strides = shape_to_strides(&shape);
/// ```
pub fn shape_to_strides(shape: &[i64]) -> Strides {
    let mut strides = vec![0; shape.len()];
    let mut size = 1;
    for i in (0..shape.len()).rev() {
        let tmp = shape[i];
        strides[i] = size;
        size *= tmp;
    }
    strides.into()
}

pub fn calculate_new_strides(last_stride: i64, shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut size = last_stride;
    for i in (0..shape.len()).rev() {
        let tmp = shape[i];
        strides[i] = size;
        size *= tmp;
    }
    strides
}

/// # Internal Function
/// Checks if the strides represent an expanded (non-default) layout.
///
/// This function determines whether a tensor's layout in memory has been expanded,
/// which typically occurs when certain dimensions are of size 1.
///
/// # Arguments
/// - `strides`: A reference to a vector containing the strides of the tensor.
///
/// # Returns
/// `true` if the tensor has an expanded layout, `false` otherwise.
///
/// # Examples
/// ```
/// use tensor_common::strides_is_expanded;
/// let strides = vec![5, 0, 1];
/// let is_expanded = strides_is_expanded(&strides);
/// ```
pub fn strides_is_expanded(strides: &[i64]) -> bool {
    let mut expanded = false;
    for i in (0..strides.len()).rev() {
        if strides[i] == 0 {
            expanded = true;
            break;
        }
    }
    expanded
}
