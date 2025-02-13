use std::panic::Location;

use crate::{tensor_base::_Tensor, Cuda};
use cudarc::driver::LaunchAsync;
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use hpt_common::{err_handler::TensorError, slice::Slice};
use hpt_traits::{
    shape_manipulate::ShapeManipulate,
    tensor::{CommonBounds, TensorCreator, TensorInfo},
};

use super::cuda_utils::{
    compile_kernel, compute_kernel_launch_config, get_include_1, get_module_name_vec,
};

/// Concatenates multiple tensors along a specified axis.
///
/// This method concatenates a list of tensors along a specified axis, with an option to retain
/// or collapse dimensions. All tensors must have the same shape except for the concatenation axis.
///
/// # Arguments
///
/// * `tensors` - A vector of references to the tensors that will be concatenated.
/// * `axis` - The axis along which the concatenation will occur. All tensors must have
///   the same shape along the non-concatenation axes.
/// * `keepdims` - A boolean flag indicating whether to retain the original dimensions of
///   the tensors in the output:
///   - If `true`, the original dimensions will be kept.
///   - If `false`, the resulting tensor will have its dimensions adjusted based on concatenation.
///
/// # Returns
///
/// This function returns a `Result` containing a new tensor that is the result of concatenating
/// the input tensors along the specified axis.
#[track_caller]
pub(crate) fn concat<T, const DEVICE: usize>(
    tensors: Vec<&_Tensor<T, Cuda, DEVICE>>,
    axis: usize,
    keepdims: bool,
) -> std::result::Result<_Tensor<T, Cuda, DEVICE>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaTypeName,
{
    let length = tensors.len();
    for i in tensors.iter() {
        for (idx, x) in tensors[0].shape().iter().enumerate() {
            if idx != axis && i.shape().len() == tensors[0].shape().len() && *x != i.shape()[idx] {
                return Err(TensorError::ConcatError(
                    axis,
                    *x as usize,
                    Location::caller(),
                ));
            } else if i.shape().len() != tensors[0].shape().len() {
                return Err(TensorError::NdimMismatched(
                    tensors[0].ndim(),
                    i.ndim(),
                    Location::caller(),
                )
                .into());
            }
        }
    }
    let mut new_shape: Vec<i64> = vec![0; tensors[0].ndim()];
    tensors.iter().for_each(|x| {
        new_shape[axis] += x.shape()[axis];
    });
    tensors[0].shape().iter().enumerate().for_each(|(i, x)| {
        if i != axis {
            new_shape[i] = *x;
        }
    });
    let new_tensor = _Tensor::<T, Cuda, DEVICE>::empty(&new_shape)?;
    let mut begin = 0;
    let mut res_slices = Vec::with_capacity(length);
    for i in tensors.iter() {
        let mut selections = vec![Slice::Full; new_shape.len()];
        selections[axis] = Slice::Range((begin, begin + i.shape()[axis]));
        begin += i.shape()[axis];
        let res_tensor = new_tensor.slice(&selections)?;
        res_slices.push(res_tensor);
    }
    let tensors = tensors
        .iter()
        .map(|x| (*x).clone())
        .collect::<Vec<_Tensor<T, Cuda, DEVICE>>>();

    let include = get_include_1::<T>();
    let module_name = get_module_name_vec("cc", &tensors);
    let map = compile_kernel(
        &module_name,
        &format!(
            "
                    {include}
                    extern \"C\" __global__ void assign({} *out, {} *inp, long long *shape, long long *strides, long long *inp_shape, long long *inp_strides, size_t ndim, size_t size)
                    {{
                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        size_t stride = blockDim.x * gridDim.x;
                        while (idx < size)
                        {{
                            long inp_amount = idx;
                            long inp_offset = 0;
                            long out_offset = 0;
                            long out_amount = idx;
                            for (int j = ndim - 1; j >= 0; j--)
                            {{
                                inp_offset += (inp_amount % inp_shape[j]) * inp_strides[j];
                                inp_amount /= inp_shape[j];
                                out_offset += (out_amount % shape[j]) * strides[j];
                                out_amount /= shape[j];
                            }}
                            out[out_offset] = inp[inp_offset];
                            idx += stride;
                        }}
                    }}",
            T::CUDA_TYPE,
            T::CUDA_TYPE,
        ),
        tensors[0].device(),
        &["assign"],
    )?;
    let kernel = tensors[0]
        .device()
        .get_func(&module_name, "assign")
        .unwrap();
    let reg_info = map.get("assign").expect("func_name not found");
    for (res, input) in res_slices.into_iter().zip(tensors.into_iter()) {
        let out_slice = res.cuda_slice();
        let inp_slice = input.cuda_slice();
        let inp_shape = new_tensor.cuda_shape()?;
        let inp_strides = new_tensor.cuda_strides()?;
        let shape = new_tensor.cuda_shape()?;
        let strides = new_tensor.cuda_strides()?;
        let cfg = compute_kernel_launch_config(res.device(), reg_info, input.size());
        unsafe {
            kernel
                .clone()
                .launch(
                    cfg,
                    (
                        out_slice,
                        inp_slice,
                        &shape,
                        &strides,
                        &inp_shape,
                        &inp_strides,
                        input.ndim() as u64,
                        input.size() as u64,
                    ),
                )
                .map_err(|e| {
                    TensorError::CudaKernelLaunchingError(
                        module_name.clone(),
                        "assign".to_string(),
                        Location::caller(),
                        e,
                    )
                })?;
        };
    }
    if keepdims {
        let mut res_shape = Vec::with_capacity(new_shape.len() + 1);
        for (idx, i) in new_shape.iter().enumerate() {
            if idx == axis {
                res_shape.push(length as i64);
                res_shape.push(*i / (length as i64));
            } else {
                res_shape.push(*i);
            }
        }
        new_tensor.reshape(res_shape)
    } else {
        Ok(new_tensor)
    }
}
