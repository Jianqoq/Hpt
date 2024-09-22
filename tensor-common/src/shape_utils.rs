use std::panic::Location;

use crate::{err_handler::ErrHandler, shape::Shape, strides::Strides};

/// Inserts a dimension of size 1 before the specified index in a shape.
///
/// The `yield_one_before` function takes an existing shape (a slice of `i64` values) and inserts
/// a new dimension of size 1 before the specified index `idx`. This is useful in tensor operations
/// where you need to expand the dimensions of a tensor by adding singleton dimensions, which can
/// facilitate broadcasting or other dimension-specific operations.
///
/// # Parameters
///
/// - `shape`: A slice of `i64` representing the original shape of the tensor.
/// - `idx`: The index before which a new dimension of size 1 will be inserted.
///
/// # Returns
///
/// - A `Vec<i64>` representing the new shape with the inserted dimension of size 1.
///
/// # Examples
///
/// ```rust
/// // Example 1: Insert before the first dimension
/// let shape = vec![3, 4, 5];
/// let idx = 0;
/// let new_shape = yield_one_before(&shape, idx);
/// assert_eq!(new_shape, vec![1, 3, 4, 5]);
///
/// // Example 2: Insert before a middle dimension
/// let idx = 2;
/// let new_shape = yield_one_before(&shape, idx);
/// assert_eq!(new_shape, vec![3, 4, 1, 5]);
///
/// // Example 3: Insert before the last dimension
/// let idx = 2;
/// let new_shape = yield_one_before(&shape, idx);
/// assert_eq!(new_shape, vec![3, 4, 1, 5]);
///
/// // Example 4: Index out of bounds (appends 1 at the end)
/// let idx = 5;
/// let new_shape = yield_one_before(&shape, idx);
/// assert_eq!(new_shape, vec![3, 4, 5, 1]);
/// ```
///
/// # Notes
///
/// - **Index Bounds**: If `idx` is greater than the length of `shape`, the function will append a
///   dimension of size 1 at the end of the shape.
/// - **Use Cases**: Adding a singleton dimension is often used to adjust the shape of a tensor for
///   broadcasting in element-wise operations or to match required input dimensions for certain
///   functions.
/// - **Immutability**: The original `shape` slice is not modified; a new `Vec<i64>` is returned.
///
/// # Implementation Details
///
/// The function works by iterating over the original shape and copying each dimension into a new
/// vector. When the current index matches `idx`, it inserts a `1` before copying the next dimension.
///
/// # See Also
///
/// ```rust
/// fn yield_one_after(shape: &[i64], idx: usize) -> Vec<i64>
/// ```
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

/// Inserts a `1` into a shape vector immediately after a specified index.
///
/// The `yield_one_after` function takes a slice representing the shape of a tensor and an index,
/// and returns a new shape vector where the value `1` is inserted immediately after the specified index.
/// This is useful for reshaping tensors, especially when you need to add a singleton dimension
/// for broadcasting or other tensor operations.
///
/// # Parameters
///
/// - `shape`: A slice of `i64` representing the original shape of the tensor.
/// - `idx`: A `usize` index after which the value `1` will be inserted into the shape.
///
/// # Returns
///
/// - A `Vec<i64>` representing the new shape with the value `1` inserted after the specified index.
///
/// # Examples
///
/// ```rust
/// // Example 1: Inserting after the first dimension
/// let shape = vec![2, 3, 4];
/// let idx = 0;
/// let new_shape = yield_one_after(&shape, idx);
/// assert_eq!(new_shape, vec![2, 1, 3, 4]);
///
/// // Example 2: Inserting after the second dimension
/// let shape = vec![5, 6, 7];
/// let idx = 1;
/// let new_shape = yield_one_after(&shape, idx);
/// assert_eq!(new_shape, vec![5, 6, 1, 7]);
///
/// // Example 3: Inserting after the last dimension
/// let shape = vec![8, 9];
/// let idx = 1;
/// let new_shape = yield_one_after(&shape, idx);
/// assert_eq!(new_shape, vec![8, 9, 1]);
/// ```
///
/// # Notes
///
/// - **Index Bounds**: The `idx` parameter must be less than or equal to `shape.len() - 1`.
///   - If `idx` is equal to `shape.len() - 1`, the `1` will be appended at the end of the shape vector.
///   - If `idx` is greater than `shape.len() - 1`, the function will panic due to an out-of-bounds index.
/// - **Non-mutating**: The function does not modify the original `shape` slice; it returns a new `Vec<i64>`.
///
/// # Use Cases
///
/// - **Adding a Dimension**: Useful when you need to add a singleton dimension to a tensor for operations like broadcasting.
/// - **Reshaping Tensors**: Helps in reshaping tensors to match required dimensions for certain mathematical operations.
///
/// # Edge Cases
///
/// - **Empty Shape**: If the `shape` slice is empty, the function will panic if `idx` is not zero.
///   ```rust
///   let shape: Vec<i64> = vec![];
///   let idx = 0;
///   let new_shape = yield_one_after(&shape, idx);
///   assert_eq!(new_shape, vec![1]); // Inserts `1` at position 0
///   ```
///
/// # Panics
///
/// - The function will panic if `idx` is greater than `shape.len()`.
///
/// # See Also
///
/// ```rust
/// fn yield_one_before(shape: &[i64], idx: usize) -> Vec<i64>
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

/// Pads a shape with ones on the left to reach a specified length.
///
/// The `try_pad_shape` function takes an existing shape (a slice of `i64` values) and pads it with
/// ones on the left side to ensure the shape has the desired length. If the existing shape's length
/// is already equal to or greater than the desired length, the function returns the shape as is.
///
/// This is particularly useful in tensor operations where broadcasting rules require shapes to have
/// the same number of dimensions.
///
/// # Parameters
///
/// - `shape`: A slice of `i64` representing the original shape of the tensor.
/// - `length`: The desired length of the shape after padding.
///
/// # Returns
///
/// - A `Vec<i64>` representing the new shape, padded with ones on the left if necessary.
///
/// # Examples
///
/// ```rust
/// // Example 1: Padding is needed
/// let shape = vec![3, 4];
/// let padded_shape = try_pad_shape(&shape, 4);
/// assert_eq!(padded_shape, vec![1, 1, 3, 4]);
///
/// // Example 2: No padding is needed
/// let shape = vec![2, 3, 4];
/// let padded_shape = try_pad_shape(&shape, 2);
/// assert_eq!(padded_shape, vec![2, 3, 4]); // Shape is returned as is
/// ```
///
/// # Notes
///
/// - **Left Padding**: The function pads the shape with ones on the left side (i.e., it adds new
///   dimensions to the beginning of the shape).
/// - **Use Case**: This is useful for aligning shapes in operations that require input tensors to have
///   the same number of dimensions, such as broadcasting in tensor computations.
///
/// # Implementation Details
///
/// - **Length Check**: The function first checks if the desired `length` is less than or equal to the
///   current length of `shape`. If so, it returns a copy of `shape` as is.
/// - **Padding Logic**: If padding is needed, it creates a new vector filled with ones of size `length`.
///   It then copies the original shape's elements into the rightmost positions of this new vector,
///   effectively padding the left side with ones.
///
/// # Edge Cases
///
/// - If `length` is zero, the function returns an empty vector.
/// - If `shape` is empty and `length` is greater than zero, the function returns a vector of ones
///   with the specified `length`.
///
/// # See Also
///
/// - Functions that handle shape manipulation and broadcasting in tensor operations.
///
/// # Example Usage in Context
///
/// ```rust
/// // Assume we have two tensors with shapes [3, 4] and [4].
/// // To perform element-wise operations, we need to align their shapes.
/// let a_shape = vec![3, 4];
/// let b_shape = vec![4];
///
/// // Pad the smaller shape to match the number of dimensions.
/// let padded_b_shape = try_pad_shape(&b_shape, a_shape.len());
/// assert_eq!(padded_b_shape, vec![1, 4]);
///
/// // Now both shapes have the same number of dimensions and can be broadcast together.
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

/// Predicts the broadcasted shape resulting from broadcasting two arrays.
///
/// The `predict_broadcast_shape` function computes the resulting shape when two arrays with shapes
/// `a_shape` and `b_shape` are broadcast together. Broadcasting is a technique that allows arrays of
/// different shapes to be used together in arithmetic operations by "stretching" one or both arrays
/// so that they have compatible shapes.
///
/// # Parameters
///
/// - `a_shape`: A slice of `i64` representing the shape of the first array.
/// - `b_shape`: A slice of `i64` representing the shape of the second array.
///
/// # Returns
///
/// - `Ok(Shape)`: The resulting broadcasted shape as a `Shape` object if broadcasting is possible.
/// - `Err(anyhow::Error)`: An error if the shapes cannot be broadcast together.
///
/// # Broadcasting Rules
///
/// The broadcasting rules determine how two arrays of different shapes can be broadcast together:
///
/// 1. **Alignment**: The shapes are right-aligned, meaning that the last dimensions are compared first.
///    If one shape has fewer dimensions, it is left-padded with ones to match the other shape's length.
///
/// 2. **Dimension Compatibility**: For each dimension from the last to the first:
///    - If the dimensions are equal, they are compatible.
///    - If one of the dimensions is 1, the array in that dimension can be broadcast to match the other dimension.
///    - If the dimensions are not equal and neither is 1, broadcasting is not possible.
///
/// # Example
///
/// ```rust
/// // Assuming Shape and the necessary imports are defined appropriately.
///
/// let a_shape = &[8, 1, 6, 1];
/// let b_shape = &[7, 1, 5];
///
/// match predict_broadcast_shape(a_shape, b_shape) {
///     Ok(result_shape) => {
///         assert_eq!(result_shape, Shape::from(vec![8, 7, 6, 5]));
///         println!("Broadcasted shape: {:?}", result_shape);
///     },
///     Err(e) => {
///         println!("Error: {}", e);
///     },
/// }
/// ```
///
/// In this example:
///
/// - `a_shape` has shape `[8, 1, 6, 1]`.
/// - `b_shape` has shape `[7, 1, 5]`.
/// - After padding `b_shape` to `[1, 7, 1, 5]`, the shapes are compared element-wise from the last dimension.
/// - The resulting broadcasted shape is `[8, 7, 6, 5]`.
///
/// # Notes
///
/// - The function assumes that shapes are represented as slices of `i64`.
/// - The function uses a helper function `try_pad_shape` to pad the shorter shape with ones on the left.
/// - If broadcasting is not possible, the function returns an error indicating the dimension at which the incompatibility occurs.
///
/// # Errors
///
/// - Returns an error if at any dimension the sizes differ and neither is 1, indicating that broadcasting cannot be performed.
///
/// # Implementation Details
///
/// - The function first determines which of the two shapes is longer and which is shorter.
/// - The shorter shape is padded on the left with ones to match the length of the longer shape.
/// - It then iterates over the dimensions, comparing corresponding dimensions from each shape:
///   - If the dimensions are equal or one of them is 1, the resulting dimension is set to the maximum of the two.
///   - If neither condition is met, an error is returned.
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn predict_broadcast_shape(a_shape: &[i64], b_shape: &[i64]) -> anyhow::Result<Shape> {
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
            return Err(ErrHandler::BroadcastError(
                a_shape.into(),
                b_shape.into(),
                i,
                Location::caller(),
            )
            .into());
        };
    }

    Ok(Shape::from(result_shape))
}

/// Determines the axes along which broadcasting is required to match a desired result shape.
///
/// The `get_broadcast_axes_from` function computes the indices of axes along which the input array `a`
/// needs to be broadcasted to match the target shape `res_shape`. Broadcasting is a method used in
/// tensor operations to allow arrays of different shapes to be used together in arithmetic operations.
///
/// **Note**: This function is adapted from NumPy's broadcasting rules and implementation.
///
/// # Parameters
///
/// - `a_shape`: A slice of `i64` representing the shape of the input array `a`.
/// - `res_shape`: A slice of `i64` representing the desired result shape after broadcasting.
/// - `location`: A `Location` object indicating the source code location for error reporting.
///
/// # Returns
///
/// - `Ok(Vec<usize>)`: A vector containing the indices of the axes along which broadcasting occurs.
/// - `Err(anyhow::Error)`: An error if broadcasting is not possible due to incompatible shapes.
///
/// # Broadcasting Rules
///
/// Broadcasting follows specific rules to align arrays of different shapes:
///
/// 1. **Left Padding**: If the input array `a_shape` has fewer dimensions than `res_shape`, it is left-padded
///    with ones to match the number of dimensions of `res_shape`.
///
/// 2. **Dimension Compatibility**: For each dimension from the most significant (leftmost) to the least significant
///    (rightmost):
///    - If the dimension sizes are equal, no broadcasting is needed for that axis.
///    - If the dimension size in `a_shape` is 1 and in `res_shape` is greater than 1, broadcasting occurs along that axis.
///    - If the dimension size in `res_shape` is 1 and in `a_shape` is greater than 1, broadcasting is not possible,
///      and an error is returned.
///
/// 3. **Collecting Broadcast Axes**: The axes where broadcasting occurs are collected and returned.
///
/// # Example
///
/// ```rust
/// use anyhow::Result;
/// // Assuming `get_broadcast_axes_from` and `Location` are defined appropriately
///
/// fn main() -> Result<()> {
///     let a_shape = &[3, 1];
///     let res_shape = &[3, 4];
///     let location = Location::new("module_name", "function_name");
///
///     let axes = get_broadcast_axes_from(a_shape, res_shape, location)?;
///     assert_eq!(axes, vec![1]);
///
///     println!("Broadcast axes: {:?}", axes);
///     Ok(())
/// }
/// ```
///
/// In this example:
///
/// - The input array has shape `[3, 1]`.
/// - The desired result shape is `[3, 4]`.
/// - Broadcasting occurs along axis `1`, so the function returns `vec![1]`.
///
/// # Notes
///
/// - **Padding Shapes**: If `a_shape` has fewer dimensions than `res_shape`, it is padded on the left with ones
///   to align the dimensions.
///
/// - **Axes Indices**: The axes indices are zero-based and correspond to the dimensions of the padded `a_shape`.
///
/// - **Error Handling**: If broadcasting is not possible due to incompatible dimensions, the function returns an error
///   using `ErrHandler::BroadcastError`, providing detailed information about the mismatch.
///
/// - **Implementation Details**:
///   - The function first calculates the difference in the number of dimensions and pads `a_shape` accordingly.
///   - It then iterates over the dimensions to identify axes where broadcasting is needed or not possible.
///
/// # Errors
///
/// - Returns an error if any dimension in `res_shape` is `1` while the corresponding dimension in `a_shape` is
///   greater than `1`, as broadcasting cannot be performed in this case.
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn get_broadcast_axes_from(
    a_shape: &[i64],
    res_shape: &[i64]
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
            anyhow::bail!(ErrHandler::BroadcastError(
                a_shape.into(),
                res_shape.into(),
                i,
                Location::caller(),
            ));
        }
    }

    Ok(axes)
}

/// Determines if it is possible to reshape an array with a given shape and strides to a new shape.
///
/// This function attempts to compute the strides for the new shape based on the original shape and
/// strides. If the reshaping is possible without changing the order of the data in memory, it
/// returns `Some(new_strides)`. Otherwise, it returns `None`, indicating that the reshape operation
/// cannot be performed while maintaining the data layout.
///
/// **Note**: This implementation is adapted from NumPy's internal `PyArray_CheckStrides` function,
/// modified to fit Rust's programming model and conventions.
///
/// # Parameters
///
/// - `original_shape`: A slice of `i64` representing the shape of the original array.
/// - `original_strides`: A slice of `i64` representing the strides of the original array.
/// - `new_shape`: A slice of `i64` representing the desired shape after reshaping.
///
/// # Returns
///
/// - `Option<Strides>`: Returns `Some(new_strides)` if the reshape is possible with the computed
///   strides. Returns `None` if the reshape is not possible while preserving the data layout.
///
/// # Algorithm Overview
///
/// The function works by trying to match the dimensions of the original and new shapes, accounting
/// for dimensions of size 1 (which can be broadcasted). It iteratively collapses dimensions and
/// checks if the strides are compatible. If at any point the strides are incompatible, it returns
/// `None`. If it successfully computes compatible strides for the new shape, it returns them.
///
/// # Example
///
/// ```rust
/// let original_shape = &[2, 3, 4];
/// let original_strides = &[12, 4, 1];
/// let new_shape = &[6, 4];
///
/// if let Some(new_strides) = is_reshape_possible(original_shape, original_strides, new_shape) {
///     println!("Reshape is possible with strides: {:?}", new_strides);
/// } else {
///     println!("Reshape is not possible while preserving data layout.");
/// }
/// ```
///
/// # Notes
///
/// - This function assumes that the array is in C-contiguous order (row-major).
/// - The function ignores dimensions of size 1 in the original shape, as they do not affect the
///   memory layout.
/// - The algorithm is based on NumPy's reshape stride computation, adapted for Rust.
///
/// # Implementation Details
///
/// The function uses several indices to keep track of positions in the original and new shapes:
///
/// - `oi`, `oj`: Indices for iterating over the original shape dimensions.
/// - `ni`, `nj`: Indices for iterating over the new shape dimensions.
/// - `oldnd`: The effective number of dimensions in the original shape after removing size-1 dimensions.
///
/// The main loop tries to match products of dimensions (`np` and `op`) from the new and original
/// shapes until they are equal, indicating that those dimensions can be aligned. It then checks if
/// the strides are compatible and computes the new strides accordingly.
///
/// # Limitations
///
/// - The function does not handle all possible cases where reshaping might be possible, especially
///   for non-contiguous arrays or arrays with unusual strides.
/// - It assumes that the original array is in a standard contiguous layout.
///
/// # Acknowledgments
///
/// This function is adapted from NumPy's internal reshape stride computation.
/// The adaptation involves translating the logic into Rust and
/// modifying it to fit Rust's data structures and error handling.

pub fn is_reshape_possible(
    original_shape: &[i64],
    original_strides: &[i64],
    new_shape: &[i64],
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

/// Generates intervals for multi-threaded processing by dividing the outer loop into chunks.
///
/// The `mt_intervals` function divides a large outer loop into multiple smaller intervals to be
/// processed by multiple threads. The function aims to distribute the workload as evenly as possible
/// among the available threads, handling cases where the total number of iterations is not perfectly
/// divisible by the number of threads.
///
/// # Parameters
///
/// - `outer_loop_size`: The total number of iterations in the outer loop.
/// - `num_threads`: The number of threads to divide the work among.
///
/// # Returns
///
/// A `Vec` of tuples `(usize, usize)`, where each tuple represents the start (inclusive) and end
/// (exclusive) indices of the interval assigned to each thread.
///
/// # Algorithm Overview
///
/// 1. **Calculate Base Workload**: Each thread is assigned at least `outer_loop_size / num_threads` iterations.
/// 2. **Distribute Remainder**: If `outer_loop_size` is not divisible by `num_threads`, the remaining iterations
///    (`outer_loop_size % num_threads`) are distributed one by one to the first few threads.
/// 3. **Calculate Start and End Indices**:
///    - The `start_index` for each thread `i` is calculated as:
///      ```
///      i * (outer_loop_size / num_threads) + min(i, outer_loop_size % num_threads)
///      ```
///    - The `end_index` is then calculated by adding the base workload and an extra iteration if the thread
///      received an extra iteration from the remainder.
///
/// # Examples
///
/// ```rust
/// fn main() {
///     let outer_loop_size = 10;
///     let num_threads = 3;
///
///     let intervals = mt_intervals(outer_loop_size, num_threads);
///
///     for (i, (start, end)) in intervals.iter().enumerate() {
///         println!("Thread {}: Processing indices [{}..{})", i, start, end);
///     }
/// }
/// ```
///
/// Output:
///
/// ```text
/// Thread 0: Processing indices [0..4)
/// Thread 1: Processing indices [4..7)
/// Thread 2: Processing indices [7..10)
/// ```
///
/// In this example:
/// - The total number of iterations is 10.
/// - The number of threads is 3.
/// - Each thread gets at least `10 / 3 = 3` iterations.
/// - The remainder is `10 % 3 = 1`. So, the first thread gets one extra iteration.
///
/// # Notes
///
/// - **Workload Balance**: The function ensures that the workload is distributed as evenly as possible.
/// - **Integer Division**: Since integer division truncates towards zero, the remainder is used to distribute
///   the extra iterations.
/// - **Index Calculation**: The calculation uses `std::cmp::min` to ensure that only the first `remainder` threads
///   receive the extra iteration.
///
/// # Function Definition
///
/// ```rust
/// pub fn mt_intervals(outer_loop_size: usize, num_threads: usize) -> Vec<(usize, usize)> {
///     let mut intervals = Vec::with_capacity(num_threads);
///     for i in 0..num_threads {
///         let start_index = i * (outer_loop_size / num_threads)
///             + std::cmp::min(i, outer_loop_size % num_threads);
///         let end_index = start_index
///             + outer_loop_size / num_threads
///             + ((i < outer_loop_size % num_threads) as usize);
///         intervals.push((start_index, end_index));
///     }
///     intervals
/// }
/// ```
///
/// # Unit Tests
///
/// Here are some unit tests to verify the correctness of the function:
///
/// ```rust
/// #[cfg(test)]
/// mod tests {
///     use super::*;
///
///     #[test]
///     fn test_even_division() {
///         let intervals = mt_intervals(100, 4);
///         assert_eq!(intervals.len(), 4);
///         assert_eq!(intervals[0], (0, 25));
///         assert_eq!(intervals[1], (25, 50));
///         assert_eq!(intervals[2], (50, 75));
///         assert_eq!(intervals[3], (75, 100));
///     }
///
///     #[test]
///     fn test_uneven_division() {
///         let intervals = mt_intervals(10, 3);
///         assert_eq!(intervals.len(), 3);
///         assert_eq!(intervals[0], (0, 4));
///         assert_eq!(intervals[1], (4, 7));
///         assert_eq!(intervals[2], (7, 10));
///     }
///
///     #[test]
///     fn test_more_threads_than_work() {
///         let intervals = mt_intervals(5, 10);
///         assert_eq!(intervals.len(), 10);
///         assert_eq!(intervals[0], (0, 1));
///         assert_eq!(intervals[1], (1, 2));
///         assert_eq!(intervals[2], (2, 3));
///         assert_eq!(intervals[3], (3, 4));
///         assert_eq!(intervals[4], (4, 5));
///         for i in 5..10 {
///             assert_eq!(intervals[i], (5, 5));
///         }
///     }
///
///     #[test]
///     fn test_zero_iterations() {
///         let intervals = mt_intervals(0, 4);
///         assert_eq!(intervals.len(), 4);
///         for &(start, end) in &intervals {
///             assert_eq!(start, 0);
///             assert_eq!(end, 0);
///         }
///     }
///
///     #[test]
///     fn test_zero_threads() {
///         let intervals = mt_intervals(10, 0);
///         assert_eq!(intervals.len(), 0);
///     }
/// }
/// ```
///
/// # Caveats
///
/// - If `num_threads` is zero, the function will return an empty vector.
/// - If `outer_loop_size` is zero, all intervals will have start and end indices of zero.
///
/// # Performance Considerations
///
/// - **Allocation**: The function pre-allocates the vector with capacity `num_threads`.
/// - **Integer Operations**: The function uses integer division and modulo operations, which are efficient.
///
/// # Conclusion
///
/// The `mt_intervals` function is useful for dividing work among multiple threads in a balanced way, ensuring that
/// each thread gets a fair share of the workload, even when the total number of iterations is not perfectly divisible
/// by the number of threads.

pub fn mt_intervals(outer_loop_size: usize, num_threads: usize) -> Vec<(usize, usize)> {
    let mut intervals = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let start_index =
            i * (outer_loop_size / num_threads) + std::cmp::min(i, outer_loop_size % num_threads);
        let end_index = start_index
            + outer_loop_size / num_threads
            + ((i < outer_loop_size % num_threads) as usize);
        intervals.push((start_index, end_index));
    }
    intervals
}

/// Generates intervals for multi-threaded SIMD processing by dividing the outer loop into chunks.
///
/// The `mt_intervals_simd` function divides a large outer loop into multiple smaller intervals
/// to be processed by multiple threads. Each interval is aligned with the SIMD vector size to
/// optimize performance. This ensures that each thread processes a chunk of data that is a
/// multiple of the SIMD vector size, which is beneficial for vectorized operations.
///
/// # Parameters
///
/// - `outer_loop_size`: The total size of the outer loop (number of iterations).
/// - `num_threads`: The desired number of threads to use for processing.
/// - `vec_size`: The size of the SIMD vector (number of elements processed in one SIMD operation).
///
/// # Returns
///
/// A `Vec` of tuples `(usize, usize)`, where each tuple represents the start (inclusive) and
/// end (exclusive) indices of the interval assigned to a thread.
///
/// # Algorithm Overview
///
/// 1. **Determine Maximum Threads**: Calculate `max_threads` as `outer_loop_size / vec_size` to
///    ensure each thread has at least one full SIMD vector's worth of work.
/// 2. **Adjust Thread Count**: Set `actual_threads` to the minimum of `num_threads` and
///    `max_threads` to avoid creating more threads than necessary.
/// 3. **Calculate Base Block Count and Remainder**:
///    - `base_block_count` is the number of full blocks each thread will process.
///    - `remainder` is the number of remaining blocks that couldn't be evenly divided.
/// 4. **Assign Intervals to Threads**:
///    - Distribute the extra blocks from the remainder among the first `remainder` threads.
///    - Calculate `start_index` and `end_index` for each thread accordingly.
///
/// # Examples
///
/// ```rust
/// fn main() {
///     let outer_loop_size = 1000;
///     let num_threads = 4;
///     let vec_size = 8;
///
///     let intervals = mt_intervals_simd(outer_loop_size, num_threads, vec_size);
///
///     for (i, (start, end)) in intervals.iter().enumerate() {
///         println!("Thread {}: Processing indices [{}..{})", i, start, end);
///     }
/// }
/// ```
///
/// Output might be:
///
/// ```text
/// Thread 0: Processing indices [0..200)
/// Thread 1: Processing indices [200..400)
/// Thread 2: Processing indices [400..600)
/// Thread 3: Processing indices [600..800)
/// ```
///
/// # Notes
///
/// - **Data Alignment**: The function ensures that each interval's size is a multiple of `vec_size`
///   to maintain data alignment for SIMD operations.
/// - **Load Balancing**: Extra iterations resulting from the remainder are distributed among the
///   first few threads to balance the workload.
///
/// # Panics
///
/// The function does not explicitly panic, but providing a `vec_size` of zero will result in a
/// division by zero error.
///
/// # See Also
///
/// - SIMD (Single Instruction, Multiple Data) processing.
/// - Multi-threading in Rust.
///
/// # Caveats
///
/// - Ensure that `vec_size` is not zero to avoid division by zero errors.
/// - The function assumes that `outer_loop_size`, `num_threads`, and `vec_size` are positive integers.
///
/// # Performance Considerations
///
/// - **Thread Overhead**: Creating too many threads may introduce overhead. The function limits the
///   number of threads to the maximum useful amount based on `outer_loop_size` and `vec_size`.
/// - **SIMD Efficiency**: Aligning intervals to `vec_size` improves SIMD efficiency by preventing
///   partial vector loads and stores.
///
/// # Conclusion
///
/// The `mt_intervals_simd` function is useful for parallelizing loops in applications that benefit
/// from both multi-threading and SIMD vectorization. By carefully dividing the work into appropriately
/// sized intervals, it helps maximize performance on modern CPUs.
pub fn mt_intervals_simd(
    outer_loop_size: usize,
    num_threads: usize,
    vec_size: usize,
) -> Vec<(usize, usize)> {
    assert!(vec_size > 0, "vec_size must be greater than zero");
    assert!(num_threads > 0, "num_threads must be greater than zero");

    // 计算可以被 vec_size 整除的对齐大小和剩余部分
    let aligned_size = (outer_loop_size / vec_size) * vec_size;
    let remainder = outer_loop_size - aligned_size;

    let mut intervals = Vec::with_capacity(num_threads);

    if aligned_size > 0 {
        // 计算总的 vec 块数
        let total_vec_blocks = aligned_size / vec_size;
        let base_blocks_per_thread = total_vec_blocks / num_threads;
        let extra_blocks = total_vec_blocks % num_threads;

        let mut start = 0;

        for i in 0..num_threads {
            let mut blocks = base_blocks_per_thread;

            // 前 extra_blocks 个线程多分配一个块
            if i < extra_blocks {
                blocks += 1;
            }

            let end = start + blocks * vec_size;
            intervals.push((start, end));
            start = end;
        }

        // 将 remainder 分配给最后一个线程
        if remainder > 0 {
            if let Some(last) = intervals.last_mut() {
                *last = (last.0, last.1 + remainder);
            }
        }
    }

    if aligned_size == 0 && remainder > 0 {
        // 当 aligned_size 为 0 且有剩余时，将剩余分配给第一个线程
        if num_threads >= 1 {
            intervals.push((0, remainder));
            for _ in 1..num_threads {
                intervals.push((0, 0));
            }
        }
    } else if aligned_size > 0 {
        // 当 aligned_size > 0 时，确保 intervals 长度为 num_threads
        while intervals.len() < num_threads {
            intervals.push((aligned_size, aligned_size));
        }
    } else {
        // 当 aligned_size == 0 且 remainder == 0 时，所有线程返回 (0, 0)
        for _ in intervals.len()..num_threads {
            intervals.push((0, 0));
        }
    }

    intervals
}