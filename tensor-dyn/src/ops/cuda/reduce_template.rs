use crate::{
    ops::cuda::reduce_utils::{reduce_prepare, uncontiguous_reduce_prepare},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use tensor_common::err_handler::TensorError;
use tensor_traits::{CommonBounds, ShapeManipulate, TensorInfo};
use tensor_types::into_scalar::Cast;

use super::cuda_slice::CudaSlice;

/// Performs a reduction operation on a tensor using customizable functions.
///
/// The `reduce_template` function is a generic template designed to facilitate reduction operations
/// on tensors. It allows for custom reduction functions to be provided, enabling flexible
/// implementation of various reduction operations (e.g., sum, product, min, max).
/// The function handles the preparation of the tensor, including transposition and reshaping,
/// and manages multithreading to optimize performance.
///
/// # Type Parameters
///
/// - `T`: The data type of the elements in the input tensor `a`.
/// - `O`: The data type of the elements in the output tensor.
/// - `F1`: The type of the full reduction function.
/// - `F2`: The type of the non-keepdims multithreaded reduction function.
/// - `F3`: The type of the keepdims outer loop size 1 reduction function.
/// - `F4`: The type of the keepdims multithreaded reduction function.
///
/// # Parameters
///
/// - `a`: A reference to the input tensor of type `_Tensor<T>`.
/// - `axes`: A slice of `usize` specifying the axes along which the reduction is performed.
/// - `init_val`: The initial value used in the reduction operation.
/// - `keepdims`: A boolean indicating whether to keep the reduced dimensions with size 1.
/// - `init_out`: A boolean indicating whether to initialize the output tensor.
/// - `c`: An optional output tensor of type `_Tensor<O>` to store the result. If `None`, a new tensor is created.
/// - `full_reduce`: A function `F1` that performs the full reduction when the tensor is reduced to a scalar.
/// - `nkd`: A function `F2` for performing the reduction when `keepdims` is `false`.
/// - `kdo1`: A function `F3` for performing the reduction when `keepdims` is `true` and the outer loop size is 1.
/// - `kd`: A function `F4` for performing the reduction when `keepdims` is `true`.
///
/// # Constraints
///
/// - `T`: Must implement `CommonBounds` and `Cast<O>`.
/// - `O`: Must implement `CommonBounds`.
/// - `F1`: Must be a function or closure that takes a mutable reference to `O` and performs the full reduction.
/// - `F2`: Must be a function or closure with the signature `Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>)`.
/// - `F3`: Must be a function or closure with the signature `Fn(usize, usize, &_Tensor<O>)`.
/// - `F4`: Must be a function or closure with the signature `Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>)`.
///
/// # Returns
///
/// - `anyhow::Result<_Tensor<O>>`: Returns the result tensor after the reduction operation. If an error occurs during the operation, an `Err` is returned.
///
/// # Algorithm Overview
///
/// 1. **Preparation**:
///    - Calls `reduce_prepare` to prepare the tensor for reduction. This involve transposing the tensor and initializing the result tensor.
///    - Determines if the fast dimension is to be kept based on the reduce axes.
///
/// 2. **Handling Last Stride**:
///    - Retrieves the last stride of the transposed tensor to ensure that it is 1, which is necessary for efficient memory access during reduction.
///
/// 3. **Full Reduction**:
///    - If all dimensions are being reduced (`a.ndim() == axes.len()`), it performs a full reduction using `full_reduce`.
///
/// 4. **Partial Reduction**:
///    - Calculates sizes for inner and outer loops based on the shape of the tensor and the reduction axes.
///    - Determines the number of threads to use for multithreading, optimizing for the size of the result tensor and the number of available threads.
///    - Depending on whether `keep_fast_dim` is `true` or `false`, and the size of the outer loop, it selects the appropriate reduction function (`nkd`, `kdo1`, or `kd`) to perform the reduction.
///
/// 5. **Reshaping the Result**:
///    - After the reduction, reshapes the result tensor to match the expected output shape, considering whether reduced dimensions are kept.
///
/// # Usage
///
/// This function is intended to be used internally by tensor operations that require reduction, such as sum, mean, min, and max functions. By providing custom functions for the reduction operations, it allows for flexibility and optimization.
///
/// # Example
///
/// ```rust
/// // Assume we have implementations for the required functions and types.
/// // For demonstration purposes only.
/// fn main() -> anyhow::Result<()> {
///     let tensor: _Tensor<f32> = ...; // Input tensor
///     let axes = &[0]; // Axes along which to reduce
///     let init_val = 0.0f32; // Initial value for reduction
///     let keepdims = false;
///     let init_out = false;
///     let c = None; // No pre-allocated output tensor
///
///     // Define the reduction functions
///     let full_reduce = |result: &mut f32| {
///         // Implement full reduction logic here
///     };
///
///     let nkd = |num_threads, inner_loop_size, inner_loop_size_2, result: &_Tensor<f32>, transposed_tensor: &_Tensor<f32>| {
///         // Implement non-keepdims reduction logic here
///     };
///
///     let kdo1 = |num_threads, inner_loop_size, result: &_Tensor<f32>| {
///         // Implement keepdims with outer loop size 1 reduction logic here
///     };
///
///     let kd = |num_threads, inner_loop_size, inner_loop_size_2, result: &_Tensor<f32>, transposed_tensor: &_Tensor<f32>| {
///         // Implement keepdims reduction logic here
///     };
///
///     let result = reduce_template(
///         &tensor,
///         axes,
///         init_val,
///         keepdims,
///         init_out,
///         c,
///         full_reduce,
///         nkd,
///         kdo1,
///         kd,
///     )?;
///
///     // Use the result tensor
///     Ok(())
/// }
/// ```
///
/// # Notes
///
/// - **Safety**: The function uses `unsafe` code when calling `get_ptr().as_mut().unwrap()` to obtain a mutable pointer to the result data. This assumes that the pointer is valid and that it's safe to mutate the data.
/// - **Assertions**: The function asserts that the last stride of the transposed tensor is 1 to ensure efficient memory access.
/// - **Multithreading**: The function uses the `rayon` crate to determine the number of available threads and to parallelize the reduction operation.
///
/// # Error Handling
///
/// - The function returns an `anyhow::Result`, propagating any errors that occur during tensor preparation or reshaping.
/// - Errors may occur if the reduction axes are invalid, if the tensor shapes are incompatible, or if there is an issue during the reshape operation.
///
/// # Dependencies
///
/// - The function depends on several traits and types:
///     - `_Tensor<T>`: A tensor type parameterized by the data type `T`.
///     - `CommonBounds`: A trait that must be implemented by `T` and `O`.
///     - `Cast<O>`: A trait implemented by `T` to allow conversion into the output scalar type `O`.
///     - `rayon`: Used for multithreading support.
///
/// # See Also
///
/// - `reduce_prepare`: A helper function used to prepare the tensor for reduction.
/// - Reduction operations like `sum`, `mean`, `min`, `max`, which use `reduce_template` internally.
///
/// # Acknowledgments
///
/// This function provides a flexible template for reduction operations on tensors, allowing for optimized implementations of various reduction functions.
#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn contiguous_reduce_template<T, F1, F2, F4, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
    full_reduce: F1,
    nkd: F2,
    kd: F4,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<O> + DeviceRepr + CudaTypeName,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    F1: Fn(CudaSlice),
    F2: Fn(usize, usize, &_Tensor<O, Cuda, DEVICE_ID>, &_Tensor<T, Cuda, DEVICE_ID>),
    F4: Fn(usize, &_Tensor<O, Cuda, DEVICE_ID>, &_Tensor<T, Cuda, DEVICE_ID>),
{
    let mut keep_fast_dim = true;
    for axis in axes.iter() {
        if a.strides()[*axis] == 1 {
            keep_fast_dim = false;
            break;
        }
    }
    let mut fused_dims: Vec<usize> = vec![];
    let (a, axes) = if !keep_fast_dim {
        let mut consec_axes = vec![];
        let mut new_axes = axes.to_vec();
        let mut max = a.ndim() - 1;
        let mut last_removed = max;
        while max > 0 {
            if !axes.contains(&max) {
                break;
            } else {
                consec_axes.push(max);
                let removed = new_axes.remove(new_axes.iter().position(|&x| x == max).unwrap());
                last_removed = removed;
            }
            max -= 1;
        }
        new_axes.push(last_removed);
        fused_dims.extend(consec_axes.iter());
        let mut new_shape = a.shape().to_vec();
        let mut prod = 1;
        for dim in fused_dims.iter() {
            prod *= new_shape[*dim];
            new_shape.remove(*dim);
        }
        new_shape.push(prod);
        (a.reshape(&new_shape)?, new_axes)
    } else {
        (a.clone(), axes.to_vec())
    };
    let (transposed_tensor, result) = reduce_prepare(&a, &axes, init_val, init_out, c)?;

    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - axes.len() - 1]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    assert_eq!(a_last_stride, 1);
    if a.ndim() == axes.len() {
        full_reduce(result.cuda_slice());
    } else {
        let inner_loop_size = *a.shape().last().unwrap() as usize;
        let a_size = a.size();
        if !keep_fast_dim {
            let outer_loop_size = a_size / inner_loop_size;
            let inner_loop_size_2 = outer_loop_size / result.size();
            nkd(
                inner_loop_size,
                inner_loop_size_2,
                &result,
                &transposed_tensor,
            );
        } else {
            let inner_loop_size_2 = a.size() / result.size();
            kd(inner_loop_size_2, &result, &transposed_tensor);
        }
    }
    result.reshape(a.layout.reduce(axes, keepdims)?.shape())
}

/// Performs a reduction operation on a non-contiguous tensor using customizable functions.
///
/// The `uncontiguos_reduce_template` function is a generic template for performing reduction
/// operations on tensors that are not contiguous in memory. It facilitates custom reduction
/// functions, enabling flexible implementation of various reduction operations (e.g., sum, product,
/// min, max) on non-contiguous tensors. The function handles the preparation of the tensor,
/// including transposition and reshaping, and manages multithreading to optimize performance.
///
/// **Note**: This function is intended for internal use within the tensor library and assumes
/// familiarity with tensor operations and memory layouts.
///
/// # Type Parameters
///
/// - `T`: The data type of the elements in the input tensor `a`.
/// - `O`: The data type of the elements in the output tensor.
/// - `F1`: The type of the full reduction function.
/// - `F2`: The type of the non-keepdims multithreaded reduction function.
/// - `F3`: The type of the keepdims outer loop size 1 reduction function.
/// - `F4`: The type of the keepdims multithreaded reduction function.
///
/// # Parameters
///
/// - `a`: A reference to the input tensor of type `_Tensor<T>`.
/// - `axes`: A slice of `usize` specifying the axes along which the reduction is performed.
/// - `init_val`: The initial value used in the reduction operation.
/// - `keepdims`: A boolean indicating whether to keep the reduced dimensions with size 1.
/// - `init_out`: A boolean indicating whether to initialize the output tensor.
/// - `c`: An optional output tensor of type `_Tensor<O>` to store the result. If `None`, a new tensor is created.
/// - `full_reduce`: A function `F1` that performs the full reduction when the tensor is reduced to a scalar.
/// - `nkd`: A function `F2` for performing the reduction when `keepdims` is `false`.
/// - `kdo1`: A function `F3` for performing the reduction when `keepdims` is `true` and the outer loop size is 1.
/// - `kd`: A function `F4` for performing the reduction when `keepdims` is `true`.
///
/// # Constraints
///
/// - `T`: Must implement `CommonBounds` and `Cast<O>`.
/// - `O`: Must implement `CommonBounds`.
/// - `F1`: Must be a function or closure that takes a mutable reference to `O` and performs the full reduction.
/// - `F2`: Must be a function or closure with the signature `Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>)`.
/// - `F3`: Must be a function or closure with the signature `Fn(usize, usize, _Tensor<T>, &_Tensor<O>)`.
/// - `F4`: Must be a function or closure with the signature `Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>)`.
///
/// # Returns
///
/// - `anyhow::Result<_Tensor<O>>`: Returns the result tensor after the reduction operation. If an error occurs during the operation, an `Err` is returned.
///
/// # Algorithm Overview
///
/// 1. **Preparation**:
///    - Calls `uncontiguous_reduce_prepare` to prepare the tensor for reduction. This may involve transposing the tensor, adjusting strides, and initializing the result tensor.
///    - Determines whether to keep the fast dimension based on `keep_fast_dim`.
///    - Computes `transposed_shape_sub_1` by decrementing each dimension size by 1. This is used for indexing calculations.
///
/// 2. **Handling Full Reduction**:
///    - If all dimensions are being reduced (`a.ndim() == axes.len()`), it performs a full reduction using `full_reduce`.
///
/// 3. **Partial Reduction**:
///    - Calculates sizes for inner and outer loops based on the shape of the tensor and the reduction axes.
///    - Determines the number of threads to use for multithreading, optimizing based on the size of the result tensor and the number of available threads.
///    - Depending on whether `keep_fast_dim` is `true` or `false`, and the size of the outer loop, it selects the appropriate reduction function (`nkd`, `kdo1`, or `kd`) to perform the reduction.
///    - If `keep_fast_dim` is `true` and the outer loop size is 1, it performs an additional permutation of the tensor to optimize memory access patterns.
///
/// 4. **Finalizing Result**:
///    - After the reduction, the result tensor is permuted back to the original dimension order using `permute_inv(res_perm)`.
///    - The tensor is reshaped to match the expected output shape, considering whether reduced dimensions are kept.
///
/// # Usage
///
/// This function is intended for internal use in tensor operations that require reduction on non-contiguous tensors. By providing custom functions for the reduction operations, it allows for flexibility and optimization in handling complex memory layouts.
///
/// # Example
///
/// ```rust
/// // Assume we have implementations for the required functions and types.
/// // For demonstration purposes only.
///
/// fn main() -> anyhow::Result<()> {
///     let tensor: _Tensor<f32> = ...; // Input non-contiguous tensor
///     let axes = &[0, 2]; // Axes along which to reduce
///     let init_val = 0.0f32; // Initial value for reduction
///     let keepdims = true;
///     let init_out = false;
///     let c = None; // No pre-allocated output tensor
///
///     // Define the reduction functions
///     let full_reduce = |result: &mut f32| {
///         // Implement full reduction logic here
///     };
///
///     let nkd = |num_threads, inner_loop_size, inner_loop_size_2, result: &_Tensor<f32>, transposed_tensor: &_Tensor<f32>| {
///         // Implement non-keepdims reduction logic for non-contiguous tensor here
///     };
///
///     let kdo1 = |num_threads, inner_loop_size, transposed_tensor: _Tensor<f32>, result: &_Tensor<f32>| {
///         // Implement keepdims with outer loop size 1 reduction logic here
///     };
///
///     let kd = |num_threads, inner_loop_size, inner_loop_size_2, result: &_Tensor<f32>, transposed_tensor: &_Tensor<f32>| {
///         // Implement keepdims reduction logic here
///     };
///
///     let result = uncontiguos_reduce_template(
///         &tensor,
///         axes,
///         init_val,
///         keepdims,
///         init_out,
///         c,
///         full_reduce,
///         nkd,
///         kdo1,
///         kd,
///     )?;
///
///     // Use the result tensor
///     Ok(())
/// }
/// ```
///
/// # Notes
///
/// - **Non-Contiguous Tensors**: This function is specifically designed to handle tensors that are not contiguous in memory, which requires special handling compared to contiguous tensors.
/// - **Permutation**: The function may permute the tensor dimensions to optimize memory access patterns during the reduction.
/// - **Safety**: The function uses `unsafe` code when accessing mutable pointers. It assumes that pointers are valid and that it's safe to mutate the data.
/// - **Multithreading**: Utilizes the `rayon` crate to determine the number of available threads and to parallelize the reduction operation for performance optimization.
///
/// # Error Handling
///
/// - Returns an `anyhow::Result`, propagating any errors that occur during tensor preparation, permutation, or reshaping.
/// - Errors may occur if the reduction axes are invalid, if tensor shapes are incompatible, or if there is an issue during permutation or reshape operations.
///
/// # Dependencies
///
/// - Requires the following traits and types:
///     - `_Tensor<T>`: A tensor type parameterized by the data type `T`.
///     - `CommonBounds`: A trait that must be implemented by `T` and `O`.
///     - `Cast<O>`: A trait implemented by `T` to allow conversion into the output scalar type `O`.
///     - `rayon`: Used for multithreading support.
///
/// # See Also
///
/// - `uncontiguous_reduce_prepare`: A helper function used to prepare the non-contiguous tensor for reduction.
/// - Reduction operations like `sum`, `mean`, `min`, `max`, which may use `uncontiguos_reduce_template` internally for non-contiguous tensors.
///
/// # Acknowledgments
///
/// This function provides a flexible template for reduction operations on non-contiguous tensors, enabling optimized implementations of various reduction functions in the context of complex memory layouts.

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn uncontiguos_reduce_template<T, F1, F2, F3, F4, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
    full_reduce: F1,
    nkd: F2,
    kdo1: F3,
    kd: F4,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<O> + DeviceRepr + CudaTypeName,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    F1: Fn(&mut O),
    F2: Fn(usize, usize, usize, &_Tensor<O, Cuda, DEVICE_ID>, &_Tensor<T, Cuda, DEVICE_ID>),
    F3: Fn(usize, usize, _Tensor<T, Cuda, DEVICE_ID>, &_Tensor<O, Cuda, DEVICE_ID>),
    F4: Fn(usize, usize, usize, &_Tensor<O, Cuda, DEVICE_ID>, &_Tensor<T, Cuda, DEVICE_ID>),
{
    let (keep_fast_dim, transposed_tensor, result, res_perm) =
        uncontiguous_reduce_prepare(a, axes, init_val, init_out, c)?;
    let mut transposed_shape_sub_1 = transposed_tensor.shape().inner().clone();
    transposed_shape_sub_1.iter_mut().for_each(|x| {
        *x -= 1;
    });

    let result_data = result.ptr();
    if a.ndim() == axes.len() {
        full_reduce(unsafe { result_data.get_ptr().as_mut().unwrap() });
    } else {
        let inner_loop_size = (if keep_fast_dim {
            transposed_tensor.shape()[a.ndim() - axes.len() - 1]
        } else {
            transposed_tensor.shape()[a.ndim() - 1]
        }) as usize;
        let a_size = a.size();
        if !keep_fast_dim {
            let outer_loop_size = a_size / inner_loop_size;
            let inner_loop_size_2 = outer_loop_size / result.size();
            let num_threads = if result.size() < rayon::current_num_threads() {
                result.size()
            } else {
                rayon::current_num_threads()
            };
            nkd(
                num_threads,
                inner_loop_size,
                inner_loop_size_2,
                &result,
                &transposed_tensor,
            );
        } else {
            let outer_loop_size = result.size() / inner_loop_size;
            let inner_loop_size_2 = a.size() / result.size();
            if outer_loop_size == 1 {
                let num_threads = if inner_loop_size < rayon::current_num_threads() {
                    inner_loop_size
                } else {
                    rayon::current_num_threads()
                };
                let mut p = (0..a.ndim()).collect::<Vec<usize>>();
                let front = p.remove(0);
                p.push(front);
                let _a = transposed_tensor.permute(&p).unwrap();
                kdo1(num_threads, inner_loop_size, _a, &result);
            } else {
                let num_threads = if outer_loop_size < rayon::current_num_threads() {
                    outer_loop_size
                } else {
                    rayon::current_num_threads()
                };
                kd(
                    num_threads,
                    inner_loop_size,
                    inner_loop_size_2,
                    &result,
                    &transposed_tensor,
                );
            }
        }
    }
    result
        .permute_inv(res_perm)?
        .reshape(a.layout.reduce(axes, keepdims)?.shape())
}
