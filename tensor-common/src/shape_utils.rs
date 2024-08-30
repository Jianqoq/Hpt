use std::panic::Location;

use crate::{ err_handler::ErrHandler, layout::Layout, shape::Shape, strides::Strides };

/// # Internal Function
/// yield `1` before `idx` for `shape`, use when you want to `manipulate the shape of a tensor`
/// ```
/// use tensor_common::yield_one_before;
/// let shape = [10, 10, 10];
/// let new_shape = yield_one_before(&shape.to_vec(), 1);
/// assert_eq!(new_shape, [10, 1, 10, 10].to_vec());
pub fn yield_one_before(shape: &[i64], idx: usize) -> Vec<i64> {
    let mut new_shape = Vec::with_capacity(shape.len() + 1);
    for (i, s) in shape.iter().enumerate() {
        if i == idx {
            new_shape.push(1);
            new_shape.push(*s);
        } else {
            new_shape.push(*s);
        }
    }
    new_shape
}

/// # Internal Function
/// yield `1` after `idx` for `shape`, use when you want to `manipulate the shape of a tensor`
/// ```
/// use tensor_common::yield_one_after;
/// let shape = [10, 10, 10];
/// let new_shape = yield_one_after(&shape.to_vec(), 1);
/// assert_eq!(new_shape, [10, 10, 1, 10].to_vec());
/// ```
pub fn yield_one_after(shape: &[i64], idx: usize) -> Vec<i64> {
    let mut new_shape = Vec::with_capacity(shape.len() + 1);
    for (i, s) in shape.iter().enumerate() {
        if i == idx {
            new_shape.push(*s);
            new_shape.push(1);
        } else {
            new_shape.push(*s);
        }
    }
    new_shape
}

/// # Internal Function
/// Pads a given shape to a specified length by prepending ones.
///
/// If the length of the provided shape is less than the specified length,
/// ones are prepended to the shape until its length matches the specified length.
/// If the shape is already equal to or longer than the specified length, it is returned as is.
///
/// # Arguments
/// - `shape`: A reference to a vector of `i64` representing the original shape.
/// - `length`: The desired length of the shape after padding.
///
/// # Returns
/// A new `Vec<i64>` representing the padded shape.
///
/// # Examples
/// ```
/// use tensor_common::try_pad_shape;
/// let original_shape = vec![2, 3];
/// let padded_shape = try_pad_shape(&original_shape, 4);
/// assert_eq!(padded_shape, vec![1, 1, 2, 3]);
/// ```
pub fn try_pad_shape(shape: &[i64], length: usize) -> Vec<i64> {
    // If the current shape length is already equal or greater, return it as is.
    if length <= shape.len() {
        return shape.to_vec();
    }

    // Otherwise, create a new shape vector with ones and overlay the existing shape on it.
    let mut ret = vec![1; length];
    for (existing, new) in shape.iter().rev().zip(ret.iter_mut().rev()) {
        *new = *existing;
    }

    ret
}

/// pad shape to the shortter one, this is used for prepareing for matmul broadcast.
///
/// possibly we can make it works in more generic cases not only matmul
pub fn compare_and_pad_shapes(a_shape: &[i64], b_shape: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let len_diff = i64::abs((a_shape.len() as i64) - (b_shape.len() as i64)) as usize;
    let (longer, shorter) = if a_shape.len() > b_shape.len() {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };

    let mut padded_shorter = vec![1; len_diff];
    padded_shorter.extend_from_slice(shorter);
    (longer.to_vec(), padded_shorter)
}

/// # Internal Function
/// Predicts the shape resulting from broadcasting two shapes.
///
/// Broadcasting allows operations to be performed on arrays of different shapes
/// in a way that the smaller array is "stretched" to match the size of the larger array.
///
/// # Arguments
/// - `a_shape`: An `Arc<Vec<i64>>` representing the first shape.
/// - `a_size`: The size of the first shape.
/// - `b_shape`: An `Arc<Vec<i64>>` representing the second shape.
/// - `b_size`: The size of the second shape.
///
/// # Returns
/// A `Result<Arc<Vec<i64>>>` representing the shape after broadcasting, or an error if
/// the shapes cannot be broadcast together.
///
/// # Errors
/// Returns an error if the shapes cannot be broadcast together.
///
/// # Examples
/// ```
/// use std::sync::Arc;
/// use tensor_common::predict_broadcast_shape;
/// let shape1 = Arc::new(vec![2, 3]);
/// let shape2 = Arc::new(vec![1, 3]);
/// let broadcasted_shape = predict_broadcast_shape(&shape1, &shape2).unwrap();
/// assert_eq!(*broadcasted_shape, vec![2, 3]);
/// ```
pub fn predict_broadcast_shape(
    a_shape: &[i64],
    b_shape: &[i64],
    location: &'static Location<'static>
) -> anyhow::Result<Shape> {
    let (longer, shorter) = if a_shape.len() >= b_shape.len() {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };

    let padded_shorter = try_pad_shape(shorter, longer.len());
    let mut result_shape = vec![0; longer.len()];

    for (i, (&longer_dim, &shorter_dim)) in longer.iter().zip(&padded_shorter).enumerate() {
        result_shape[i] = if longer_dim == shorter_dim || shorter_dim == 1 {
            longer_dim
        } else if longer_dim == 1 {
            shorter_dim
        } else {
            return Err(
                ErrHandler::BroadcastError(a_shape.into(), b_shape.into(), i, location).into()
            );
        };
    }

    Ok(Shape::from(result_shape))
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub fn predict_broadcast_strides<T: Into<Layout>>(
    brocasted_shape: &[i64],
    original_layout: T
) -> Strides {
    let original_layout: Layout = original_layout.into();
    let brocasted_size = brocasted_shape.iter().product::<i64>();
    let original_size = original_layout.size();

    // if true, it is brocasted
    if brocasted_size > original_size {
        let shape = try_pad_shape(original_layout.shape(), brocasted_shape.len());
        let axes_to_broadcast = get_broadcast_axes_from(
            &shape,
            brocasted_shape,
            Location::caller()
        ).expect("Cannot broadcast shapes");

        let mut new_strides = vec![0; brocasted_shape.len()];
        new_strides
            .iter_mut()
            .rev()
            .zip(original_layout.strides().iter().rev())
            .for_each(|(a, b)| {
                *a = *b;
            });
        for &axis in axes_to_broadcast.iter() {
            assert_eq!(shape[axis], 1);
            new_strides[axis] = 0;
        }
        new_strides.into()
    } else {
        ErrHandler::check_size_match(original_layout.size(), brocasted_size).unwrap();
        if let Some(new_strides) = original_layout.is_reshape_possible(brocasted_shape) {
            new_strides
        } else {
            ErrHandler::IterInplaceReshapeError(
                brocasted_shape.into(),
                original_layout.shape().clone(),
                original_layout.strides().clone(),
                Location::caller()
            );
            unreachable!()
        }
    }
}

pub fn get_broadcast_axes_from(
    a_shape: &[i64],
    res_shape: &[i64],
    location: &'static Location<'static>
) -> anyhow::Result<Vec<usize>> {
    assert!(a_shape.len() <= res_shape.len());

    let padded_a = try_pad_shape(a_shape, res_shape.len());

    let mut axes = Vec::new();
    let padded_axes = (0..res_shape.len() - a_shape.len()).collect::<Vec<usize>>();
    for i in padded_axes.iter() {
        axes.push(*i);
    }

    for (i, (&res_dim, &a_dim)) in res_shape.iter().zip(&padded_a).enumerate() {
        if a_dim == 1 && res_dim != 1 && !padded_axes.contains(&i) {
            axes.push(i);
        } else if res_dim == 1 && a_dim != 1 {
            anyhow::bail!(
                ErrHandler::BroadcastError(a_shape.into(), res_shape.into(), i, location)
            );
        }
    }

    Ok(axes)
}

/// # Internal Function
/// Currently used in backward propagation.
///
/// # Arguments
/// `a_shape`: Forward operand shape.
/// `b_shape`: Forward output shape.
///
/// # Returns
/// (`a_axes`, `b_axes`), `a_axes` are the axes to reduce along in `a_shape`, `b_axes` normally is empty.
///
/// # Examples
/// ```
/// use tensor_common::detect_broadcast_axes;
/// let a_shape = [2, 3, 4];
/// let b_shape = [2, 3, 4];
/// let (a_axes, b_axes) = detect_broadcast_axes(&a_shape, &b_shape);
/// assert_eq!(a_axes, vec![]);
/// assert_eq!(b_axes, vec![]);
/// ```
pub fn detect_broadcast_axes(a_shape: &[i64], b_shape: &[i64]) -> (Vec<usize>, Vec<usize>) {
    let (longer, shorter) = if a_shape.len() >= b_shape.len() {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };

    let padded_shorter = try_pad_shape(shorter, longer.len());
    let mut a_axes = Vec::new();
    let mut b_axes = Vec::new();

    for (i, (&longer_dim, &shorter_dim)) in longer.iter().zip(&padded_shorter).enumerate() {
        if longer_dim == shorter_dim {
            continue;
        } else if longer_dim == 1 {
            a_axes.push(i);
        } else if shorter_dim == 1 {
            b_axes.push(i);
        }
    }
    if a_shape.len() >= b_shape.len() {
        (a_axes, b_axes)
    } else {
        (b_axes, a_axes)
    }
}

/// # Internal Function
/// Checks if a shape can be broadcast to match another shape.
///
/// # Arguments
/// - `a_shape`: A slice of `i64` representing the original shape.
/// - `res_shape`: A slice of `i64` representing the target shape.
///
/// # Returns
/// A `bool` representing the new shape if broadcastable.
///
/// # Examples
/// ```
/// use tensor_common::is_broadcastable;
/// let shape1 = [2, 1];
/// let shape2 = [2, 3];
/// let new_shape = is_broadcastable(&shape1, &shape2);
/// assert_eq!(new_shape, true);
/// ```
pub fn is_broadcastable(a_shape: &[i64], b_shape: &[i64]) -> bool {
    let (longer, shorter) = if a_shape.len() >= b_shape.len() {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };

    let padded_shorter = try_pad_shape(shorter, longer.len());

    for (&longer_dim, &shorter_dim) in longer.iter().zip(&padded_shorter) {
        if !(longer_dim == shorter_dim || longer_dim == 1 || shorter_dim == 1) {
            return false;
        }
    }

    true
}

/// # Internal Function
/// Predicts the shape resulting from reducing a shape along specified axes.
///
/// # Arguments
/// - `shape`: A slice of `i64` representing the original shape.
/// - `axes`: A slice of `usize` representing the axes to reduce along.
///
/// # Returns
/// (shape_with_zero, res_shape)
/// - `shape_with_zero`: A `Vec<i64>` representing the shape with zeros at the axes to reduce along.
/// - `res_shape`: A `Vec<i64>` representing the shape after reducing.
///
/// # Examples
/// ```
/// use tensor_common::predict_reduce_shape;
/// let shape = [2, 3, 4];
/// let axes = [0, 2];
/// let (shape_with_zero, res_shape) = predict_reduce_shape(&shape, &axes);
/// assert_eq!(shape_with_zero, [0, 3, 0]);
/// assert_eq!(res_shape, [3]);
/// ```
pub fn predict_reduce_shape(shape: &[i64], axes: &[usize]) -> (Vec<i64>, Vec<i64>) {
    let mut shape_with_zero = shape.to_vec();
    for axis in axes.iter() {
        shape_with_zero[*axis] = 0;
    }
    let mut res_shape = Vec::with_capacity(shape.len() - axes.len());
    shape_with_zero.iter().for_each(|x| {
        if *x != 0 {
            res_shape.push(*x)
        }
    });
    if res_shape.is_empty() {
        res_shape.push(1);
    }
    (shape_with_zero, res_shape)
}

/// # Internal Function
/// Predicts the shape resulting from reducing a shape along specified axes.
///
/// for axes that are to reduce along, the shape will be `0`
///
/// # Arguments
/// - `shape`: A slice of `i64` representing the original shape.
/// - `axes`: A slice of `usize` representing the axes to reduce along.
///
/// # Returns
/// A `Vec<i64>` representing the shape after reducing.
///
/// # Examples
/// ```
/// use tensor_common::predict_reduce_shape_with_zero;
/// let shape = [2, 3, 4];
/// let axes = [0, 2];
/// let res_shape = predict_reduce_shape_with_zero(&shape, &axes);
/// assert_eq!(res_shape, [0, 3, 0]);
/// ```
pub fn predict_reduce_shape_with_zero(shape: &[i64], axes: &[usize]) -> Vec<i64> {
    let mut shape_with_zero = shape.to_vec();
    for axis in axes.iter() {
        shape_with_zero[*axis] = 0;
    }
    shape_with_zero
}

// code is rewrite from numpy
pub fn is_reshape_possible(
    original_shape: &[i64],
    original_strides: &[i64],
    new_shape: &[i64]
) -> Option<Strides> {
    let mut new_strides = vec![0; new_shape.len()];
    let mut old_strides = vec![0; original_shape.len()];
    let mut old_shape = vec![0; original_shape.len()];

    let mut oi = 0;
    let mut oj = 1;
    let mut ni = 0;
    let mut nj = 1;

    let mut oldnd = 0;

    for i in 0..original_shape.len() {
        if original_shape[i] != 1 {
            old_shape[oldnd] = original_shape[i];
            old_strides[oldnd] = original_strides[i];
            oldnd += 1;
        }
    }

    while ni < new_shape.len() && oi < oldnd {
        let mut np = new_shape[ni];
        let mut op = old_shape[oi];

        while np != op {
            if np < op {
                np *= new_shape[nj];
                nj += 1;
            } else {
                op *= old_shape[oj];
                oj += 1;
            }
        }

        for i in oi..oj - 1 {
            if old_strides[i] != old_shape[i + 1] * old_strides[i + 1] {
                return None;
            }
        }

        new_strides[nj - 1] = old_strides[oj - 1];
        for i in (ni + 1..nj).rev() {
            new_strides[i - 1] = new_strides[i] * new_shape[i];
        }

        ni = nj;
        nj += 1;
        oi = oj;
        oj += 1;
    }

    let last_stride = if ni >= 1 { new_strides[ni - 1] } else { 1 };

    for i in ni..new_shape.len() {
        new_strides[i] = last_stride;
    }

    Some(new_strides.into())
}

/// # Internal Function
/// Predicts the shape resulting from reducing a shape along specified axes.
///
/// for axes that are to reduce along, the shape will be `1`
///
/// # Arguments
/// - `shape`: A slice of `i64` representing the original shape.
/// - `axes`: A slice of `usize` representing the axes to reduce along.
///
/// # Returns
/// A `Vec<i64>` representing the shape after reducing.
///
/// # Examples
/// ```
/// use tensor_common::predict_reduce_shape_with_one;
/// let shape = [2, 3, 4];
/// let axes = [0, 2];
/// let res_shape = predict_reduce_shape_with_one(&shape, &axes);
/// assert_eq!(res_shape, [1, 3, 1]);
/// ```
pub fn predict_reduce_shape_with_one(shape: &[i64], axes: &[usize]) -> Vec<i64> {
    let mut shape_with_zero = shape.to_vec();
    for axis in axes.iter() {
        shape_with_zero[*axis] = 1;
    }
    shape_with_zero
}

/// # Internal Function
/// do splitting on the outer loop of a tensor for multi-threading
///
/// # Arguments
/// - `outer_loop_size`: the size of the outer loop
/// - `num_threads`: the number of threads
///
/// # Returns
/// `Vec<(start, end)>`
///
/// # Examples
/// ```
/// use tensor_common::mt_intervals;
/// let intervals = mt_intervals(10, 3);
/// assert_eq!(intervals, vec![(0, 4), (4, 8), (8, 10)]);
/// ```
pub fn mt_intervals(outer_loop_size: usize, num_threads: usize) -> Vec<(usize, usize)> {
    let mut intervals = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let start_index =
            i * (outer_loop_size / num_threads) + std::cmp::min(i, outer_loop_size % num_threads);
        let end_index =
            start_index +
            outer_loop_size / num_threads +
            ((i < outer_loop_size % num_threads) as usize);
        intervals.push((start_index, end_index));
    }
    intervals
}

pub fn mt_intervals_simd(
    outer_loop_size: usize,
    num_threads: usize,
    vec_size: usize
) -> Vec<(usize, usize)> {
    let mut intervals = Vec::with_capacity(num_threads);

    let max_threads = outer_loop_size / vec_size;
    let actual_threads = num_threads.min(max_threads);

    let vec_block_size = vec_size;
    let base_block_count = outer_loop_size / vec_block_size / actual_threads;
    let remainder = outer_loop_size % (vec_block_size * actual_threads);

    for i in 0..actual_threads {
        let start_index = i * base_block_count * vec_block_size + vec_size.min(i * remainder);
        let end_index = if i == actual_threads - 1 {
            outer_loop_size
        } else {
            start_index + base_block_count * vec_block_size + vec_size.min(remainder)
        };
        intervals.push((start_index, end_index));
    }
    intervals
}
pub fn is_broadcast(a_shape: &[i64], b_shape: &[i64]) -> bool {
    let mut a_shape = a_shape.to_vec();
    let mut b_shape = b_shape.to_vec();
    if a_shape.len() < b_shape.len() {
        std::mem::swap(&mut a_shape, &mut b_shape);
    }
    let mut a_shape = a_shape.iter().rev();
    let mut b_shape = b_shape.iter().rev();
    loop {
        match (a_shape.next(), b_shape.next()) {
            (Some(&a), Some(&b)) => {
                if a == b || b == 1 {
                    continue;
                } else {
                    return false;
                }
            }
            (Some(&a), None) => {
                if a == 1 {
                    continue;
                } else {
                    return false;
                }
            }
            (None, None) => {
                return true;
            }
            (None, Some(&b)) => {
                if b == 1 {
                    continue;
                } else {
                    return false;
                }
            }
        }
    }
}
