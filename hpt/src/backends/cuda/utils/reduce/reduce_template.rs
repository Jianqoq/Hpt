use crate::backends::common::reduce::{
    get_fast_dim_size, get_new_reduce_axes, get_new_shape, is_keep_fast_dim, split_groups_by_axes,
};
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
    let groups = a.layout.coalesce_dims();
    let new_groups = split_groups_by_axes(&groups, axes);
    let new_shape = get_new_shape(&new_groups, a.shape());
    let original_ptr = a.ptr();
    let a = a.reshape(&new_shape)?;
    let new_ptr = a.ptr();
    assert_eq!(original_ptr.ptr, new_ptr.ptr);
    let axes = get_new_reduce_axes(new_groups, axes);
    let keep_fast_dim = is_keep_fast_dim(a.strides(), &axes);

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
        let a_size = a.size();
        if !keep_fast_dim {
            let inner_loop_size = get_fast_dim_size(&a.shape(), &a.strides(), &axes) as usize;
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
