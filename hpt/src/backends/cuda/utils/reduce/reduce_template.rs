use crate::backends::cuda::cuda_slice::CudaSlice;
use crate::{
    backend::Cuda, backends::cuda::utils::reduce::reduce_utils::reduce_prepare,
    tensor_base::_Tensor,
};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::ops::shape_manipulate::ShapeManipulate;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::CudaType;

#[track_caller]
pub(crate) fn contiguous_reduce_template<T, F1, F2, F4, O, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, Cuda, DEVICE_ID, Al>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cuda, DEVICE_ID, Al>>,
    full_reduce: F1,
    nkd: F2,
    kd: F4,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID, Al>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType,
    O: CommonBounds + DeviceRepr + CudaType,
    F1: Fn(CudaSlice),
    F2: Fn(
        usize,
        usize,
        &_Tensor<O, Cuda, DEVICE_ID, Al>,
        &_Tensor<T, Cuda, DEVICE_ID, Al>,
        &[usize],
    ),
    F4: Fn(usize, &_Tensor<O, Cuda, DEVICE_ID, Al>, &_Tensor<T, Cuda, DEVICE_ID, Al>, &[usize]),
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
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
        if a.ndim() == 1 {
            (a.clone(), axes.to_vec())
        } else {
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
        }
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
        let inner_loop_size = (if keep_fast_dim {
            transposed_tensor.shape()[a.ndim() - axes.len() - 1]
        } else {
            transposed_tensor.shape()[a.ndim() - 1]
        }) as usize;
        let a_size = a.size();
        if !keep_fast_dim {
            let outer_loop_size = a_size / inner_loop_size;
            let inner_loop_size_2 = outer_loop_size / result.size();
            nkd(
                inner_loop_size,
                inner_loop_size_2,
                &result,
                &transposed_tensor,
                &axes,
            );
        } else {
            let inner_loop_size_2 = a.size() / result.size();
            kd(inner_loop_size_2, &result, &transposed_tensor, &axes);
        }
    }
    result.reshape(a.layout.reduce(axes, keepdims)?.shape())
}
