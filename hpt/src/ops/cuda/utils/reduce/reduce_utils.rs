use crate::ops::common::reduce::rearrange_array;
use crate::ops::cuda::cuda_utils::load_ptx_and_get_data;
use crate::{backend::Cuda, tensor_base::_Tensor};
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchAsync;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_cudakernels::SET_VAL;
use hpt_traits::{CommonBounds, ShapeManipulate, TensorCreator, TensorInfo};
use hpt_types::dtype::CudaType;

use crate::ops::cuda::cuda_utils::compute_kernel_launch_config;

pub(crate) fn reduce_prepare<
    T: CommonBounds + DeviceRepr + CudaType,
    O: CommonBounds + DeviceRepr + CudaType,
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
        ShapeError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout())?;
        if init_out {
            let (kernel, reg_info) = load_ptx_and_get_data(
                "set_val",
                &format!("set_val_{}", O::STR),
                a.device(),
                a.device_cap(),
                &SET_VAL,
            )
            .unwrap();
            let cfg =
                compute_kernel_launch_config(out.device(), &reg_info, res_layout.size() as usize);
            unsafe {
                kernel.launch(
                    cfg,
                    (out.cuda_slice(), init_val, res_layout.size() as usize),
                )?
            };
        }
        out.reshape(res_layout.shape())?
    } else {
        let res = _Tensor::<O, Cuda, DEVICE_ID>::empty(res_layout.shape())?;
        res
    };
    Ok((a.permute(transposed_axis)?, res))
}
