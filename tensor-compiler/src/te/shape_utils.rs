use crate::halide::prime_expr::PrimeExpr;

pub fn try_pad_shape_expr(shape: &[PrimeExpr], length: usize) -> Vec<PrimeExpr> {
    // If the current shape length is already equal or greater, return it as is.
    if length <= shape.len() {
        return shape.to_vec();
    }

    // Otherwise, create a new shape vector with ones and overlay the existing shape on it.
    let mut ret = vec![1i64.into(); length];
    for (existing, new) in shape.iter().rev().zip(ret.iter_mut().rev()) {
        *new = existing.clone();
    }
    ret
}

pub fn detect_broadcast_axes_expr(
    a_shape: &[PrimeExpr],
    b_shape: &[PrimeExpr]
) -> (Vec<usize>, Vec<usize>) {
    let (longer, shorter) = if a_shape.len() >= b_shape.len() {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };

    let padded_shorter = try_pad_shape_expr(shorter, longer.len());
    let mut a_axes = Vec::new();
    let mut b_axes = Vec::new();

    for (i, (longer_dim, shorter_dim)) in longer.iter().zip(&padded_shorter).enumerate() {
        if longer_dim == shorter_dim {
            continue;
        } else if longer_dim == &(1i64).into() {
            a_axes.push(i);
        } else if shorter_dim == &(1i64).into() {
            b_axes.push(i);
        }
    }
    if a_shape.len() >= b_shape.len() {
        (a_axes, b_axes)
    } else {
        (b_axes, a_axes)
    }
}

pub fn predict_broadcast_shape_expr(
    a_shape: &[PrimeExpr],
    b_shape: &[PrimeExpr]
) -> Vec<PrimeExpr> {
    let (longer, shorter) = if a_shape.len() >= b_shape.len() {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };

    let padded_shorter = try_pad_shape_expr(shorter, longer.len());
    let mut result_shape = vec![0i64.into(); longer.len()];

    for (i, (longer_dim, shorter_dim)) in longer.iter().zip(&padded_shorter).enumerate() {
        result_shape[i] = if longer_dim == shorter_dim || shorter_dim == &(1i64).into() {
            longer_dim.clone()
        } else if longer_dim == &(1i64).into() {
            shorter_dim.clone()
        } else {
            panic!("Incompatible broadcast shapes: {:?} and {:?}", a_shape, b_shape);
        };
    }
    result_shape
}
