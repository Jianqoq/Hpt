use crate::{
    ops::cpu::reduce_utils::{reduce_prepare, uncontiguous_reduce_prepare},
    tensor_base::_Tensor,
};
use tensor_traits::{CommonBounds, ShapeManipulate, TensorInfo};
use tensor_types::into_scalar::IntoScalar;

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce_template<T, F1, F2, F3, F4, O>(
    a: &_Tensor<T>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O>>,
    full_reduce: F1,
    nkd: F2,
    kdo1: F3,
    kd: F4,
) -> anyhow::Result<_Tensor<O>>
where
    T: CommonBounds + IntoScalar<O>,
    O: CommonBounds,
    F1: Fn(&mut O),
    F2: Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>),
    F3: Fn(usize, usize, &_Tensor<O>),
    F4: Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>),
{
    let (keep_fast_dim, transposed_tensor, result) =
        reduce_prepare(a, axes, init_val, init_out, c)?;

    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - axes.len() - 1]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    assert_eq!(a_last_stride, 1);
    let result_data = result.ptr();
    if a.ndim() == axes.len() {
        full_reduce(unsafe { result_data.get_ptr().as_mut().unwrap() });
    } else {
        let inner_loop_size = *a.shape().last().unwrap() as usize;
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
                kdo1(num_threads, inner_loop_size, &result)
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
                )
            }
        }
    }
    result.reshape(a.layout.reduce(axes, keepdims)?.shape())
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn uncontiguos_reduce_template<T, F1, F2, F3, F4, O>(
    a: &_Tensor<T>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O>>,
    full_reduce: F1,
    nkd: F2,
    kdo1: F3,
    kd: F4,
) -> anyhow::Result<_Tensor<O>>
where
    T: CommonBounds + IntoScalar<O>,
    O: CommonBounds,
    F1: Fn(&mut O),
    F2: Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>),
    F3: Fn(usize, usize, _Tensor<T>, &_Tensor<O>),
    F4: Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>),
{
    let (keep_fast_dim, transposed_tensor, result, res_perm) =
        uncontiguous_reduce_prepare(a, axes, init_val, init_out, c)?;

    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - axes.len() - 1]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    assert_eq!(a_last_stride, 1);
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
                kdo1(num_threads, inner_loop_size, _a, &result)
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
                )
            }
        }
    }
    result.permute_inv(res_perm)?.reshape(a.layout.reduce(axes, keepdims)?.shape())
}
