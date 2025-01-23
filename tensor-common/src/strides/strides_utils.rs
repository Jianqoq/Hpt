use tensor_types::{convertion::Convertor, dtype::TypeCommon, into_scalar::IntoScalar};

use crate::strides::strides::Strides;

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