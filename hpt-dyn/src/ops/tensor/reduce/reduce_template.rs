use hpt_common::error::base::TensorError;
use hpt_types::dtype::DType;

use crate::{
    Tensor, current_num_threads,
    ops::tensor::reduce::reduce_utils::{
        get_fast_dim_size, get_new_reduce_axes, get_new_shape, is_keep_fast_dim, reduce_prepare,
        split_groups_by_axes,
    },
};

use super::reduce_utils::uncontiguous_reduce_prepare;

pub(crate) fn contiguous_reduce_template<F1, F2, F3, F4>(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
    full_reduce: F1,
    nkd: F2,
    kdo1: F3,
    kd: F4,
) -> std::result::Result<Tensor, TensorError>
where
    F1: Fn(usize),
    F2: Fn(usize, usize, usize, &Tensor, &Tensor),
    F3: Fn(usize, usize, &Tensor, &Tensor),
    F4: Fn(usize, usize, usize, usize, &Tensor, &Tensor),
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
    let (transposed_tensor, result) = reduce_prepare(&a, &axes, init_val, init_out, res_dtype, c)?;
    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - axes.len() - 1]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    assert_eq!(a_last_stride, 1);
    let result_data = result.ptr();
    if a.ndim() == axes.len() {
        full_reduce(result_data.ptr as usize);
    } else {
        let a_size = a.size();
        if !keep_fast_dim {
            let inner_loop_size = get_fast_dim_size(&a.shape(), &a.strides(), &axes) as usize;
            let outer_loop_size = a_size / inner_loop_size;
            let inner_loop_size_2 = outer_loop_size / result.size();
            let num_threads = result.size().min(current_num_threads());
            nkd(
                num_threads,
                inner_loop_size,
                inner_loop_size_2,
                &result,
                &transposed_tensor,
            );
        } else {
            let inner_loop_size = *a.shape().last().unwrap() as usize;
            let outer_loop_size = result.size() / inner_loop_size;
            let inner_loop_size_2 = a.size() / result.size();
            if outer_loop_size == 1 {
                let num_threads = inner_loop_size.min(current_num_threads());
                kdo1(num_threads, inner_loop_size, &result, &a);
            } else {
                let num_threads = outer_loop_size.min(current_num_threads());
                kd(
                    num_threads,
                    outer_loop_size,
                    inner_loop_size,
                    inner_loop_size_2,
                    &result,
                    &transposed_tensor,
                );
            }
        }
    }
    result.reshape(a.layout.reduce(&axes, keepdims)?.shape())
}

pub(crate) fn uncontiguos_reduce_template<F1, F2, F3, F4>(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
    full_reduce: F1,
    nkd: F2,
    kdo1: F3,
    kd: F4,
) -> std::result::Result<Tensor, TensorError>
where
    F1: Fn(usize),
    F2: Fn(usize, usize, usize, &Tensor, &Tensor),
    F3: Fn(usize, usize, Tensor, &Tensor),
    F4: Fn(usize, usize, usize, &Tensor, &Tensor),
{
    let (keep_fast_dim, transposed_tensor, result, res_perm) =
        uncontiguous_reduce_prepare(a, axes, init_val, init_out, res_dtype, c)?;
    let mut transposed_shape_sub_1 = transposed_tensor.shape().inner().clone();
    transposed_shape_sub_1.iter_mut().for_each(|x| {
        *x -= 1;
    });

    let result_data = result.ptr();
    if a.ndim() == axes.len() {
        full_reduce(result_data.ptr as usize);
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
            let num_threads = result.size().min(current_num_threads());
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
                let num_threads = inner_loop_size.min(current_num_threads());
                let mut p = (0..a.ndim() as i64).collect::<Vec<i64>>();
                let front = transposed_tensor
                    .shape()
                    .inner()
                    .iter()
                    .position(|x| *x == result.size() as i64)
                    .unwrap();
                p.remove(front);
                p.push(front as i64);
                let _a = transposed_tensor.permute(&p).unwrap();
                kdo1(num_threads, inner_loop_size, _a, &result);
            } else {
                let num_threads = outer_loop_size.min(current_num_threads());
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
        .permute_inv(&res_perm)?
        .reshape(a.layout.reduce(axes, keepdims)?.shape())
}
