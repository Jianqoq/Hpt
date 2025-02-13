use std::panic::Location;

use crate::ops::cuda::cuda_utils::load_ptx_and_get_data;
use crate::{backend::Cuda, tensor_base::_Tensor};
use cudarc::driver::LaunchAsync;
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use hpt_common::err_handler::TensorError;
use hpt_cudakernels::SET_VAL;
use hpt_traits::{CommonBounds, ShapeManipulate, TensorCreator, TensorInfo, TensorLike};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use super::cuda_utils::compute_kernel_launch_config;

pub(crate) fn rearrange_array(ndim: usize, to_reduce: &[usize]) -> Vec<usize> {
    let mut origin_order = (0..ndim).collect::<Vec<usize>>();
    let mut to_reduce = to_reduce.to_vec();
    // sort the reduce axes
    to_reduce.sort();

    // store the elements to be reduced
    let mut moved_elements = Vec::new();
    origin_order.retain(|&x| {
        if to_reduce.contains(&x) {
            moved_elements.push(x);
            false
        } else {
            true
        }
    });

    // put the reduced elements at the end
    origin_order.extend(moved_elements);

    origin_order
}

pub(crate) fn reduce_prepare<
    T: CommonBounds + DeviceRepr + CudaTypeName,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    init_out: bool,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<(_Tensor<T, Cuda, DEVICE_ID>, _Tensor<O, Cuda, DEVICE_ID>), TensorError> {
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.layout.ndim(), axes);

    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.layout.ndim() - axes.len()]
        .sort_by(|x, y| a.layout.strides()[*y].cmp(&a.layout.strides()[*x]));
    transposed_axis[a.layout.ndim() - axes.len()..]
        .sort_by(|x, y| a.layout.strides()[*y].cmp(&a.layout.strides()[*x]));

    let res_layout = a.layout.reduce(axes, false)?;
    let res = if let Some(out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        TensorError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout())?;
        if init_out {
            let (kernel, reg_info) = load_ptx_and_get_data(
                "set_val",
                &format!("set_val_{}", O::ID),
                a.device(),
                a.device_cap(),
                &SET_VAL,
            )
            .unwrap();
            let cfg =
                compute_kernel_launch_config(out.device(), &reg_info, res_layout.size() as usize);
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (out.cuda_slice(), init_val, res_layout.size() as usize),
                    )
                    .map_err(|e| {
                        TensorError::CudaKernelLaunchingError(
                            "set_val".to_string(),
                            format!("set_val_{}", O::ID),
                            Location::caller(),
                            e,
                        )
                    })?
            };
        }
        out.reshape(res_layout.shape())?
    } else {
        let res = _Tensor::<O, Cuda, DEVICE_ID>::empty(res_layout.shape())?;
        res
    };
    Ok((a.permute(transposed_axis)?, res))
}

pub(crate) fn uncontiguous_reduce_prepare<
    T: CommonBounds + DeviceRepr + CudaTypeName,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    init_out: bool,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<
    (
        bool,
        _Tensor<T, Cuda, DEVICE_ID>,
        _Tensor<O, Cuda, DEVICE_ID>,
        Vec<usize>,
    ),
    TensorError,
> {
    let mut keep_fast_dim = true;
    for axis in axes.iter() {
        if a.strides()[*axis] == 1 {
            keep_fast_dim = false;
            break;
        }
    }
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.ndim(), axes);

    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.layout.ndim() - axes.len()]
        .sort_by(|x, y| a.layout.strides()[*y].cmp(&a.layout.strides()[*x]));
    transposed_axis[a.layout.ndim() - axes.len()..]
        .sort_by(|x, y| a.layout.strides()[*y].cmp(&a.layout.strides()[*x]));

    let res_layout = a.layout.reduce(axes, false)?;

    let mut res_permute_axes = (0..res_layout.ndim()).collect::<Vec<usize>>();
    res_permute_axes.sort_by(|a, b| transposed_axis[*a].cmp(&transposed_axis[*b]));

    let res = if let Some(mut out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        TensorError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout())?;
        if init_out {
            out.as_raw_mut().par_iter_mut().for_each(|x| {
                *x = init_val;
            });
        }
        Ok(out)
    } else {
        _Tensor::<O, Cuda, DEVICE_ID>::full(init_val, res_layout.shape())?
            .permute(&res_permute_axes)
    };
    Ok((
        keep_fast_dim,
        a.permute(transposed_axis)?,
        res?,
        res_permute_axes,
    ))
}
