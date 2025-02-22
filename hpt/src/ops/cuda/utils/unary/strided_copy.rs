use crate::ops::cuda::cuda_utils::{compile_kernel, compute_kernel_launch_config, get_include_1};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use crate::TensorInfo;
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync};
use hpt_common::error::base::TensorError;
use hpt_traits::CommonBounds;
use hpt_types::dtype::CudaType;

pub(crate) fn strided_copy<T, const DEVICE_ID: usize>(
    x: &CudaSlice<T>,
    out: &mut _Tensor<T, Cuda, DEVICE_ID>,
) -> std::result::Result<(), TensorError>
where
    T: CommonBounds + CudaType + DeviceRepr,
{
    let include = get_include_1::<T>();
    let module_name = format!("strided_copy_{}", T::CUDA_TYPE);
    let map = compile_kernel(
        &module_name,
        &format!(
            "
                {include}
                extern \"C\" __global__ void strided_copy({} *out, {} *inp, long long *shape, long long *strides, size_t ndim, size_t size)
                {{
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t stride = blockDim.x * gridDim.x;
                    while (idx < size)
                    {{
                        long long amount = idx;
                        long long offset = 0;
                        #pragma unroll
                        for (int j = ndim - 1; j >= 0; j--)
                        {{
                            offset += amount % shape[j] * strides[j];
                            amount /= shape[j];
                        }}
                        out[offset] = inp[idx];
                        idx += stride;
                    }}
                }}",
            T::CUDA_TYPE,
            T::CUDA_TYPE,
        ),
        out.device(),
        &["strided_copy"],
    )?;
    let kernel = out.device().get_func(&module_name, "strided_copy").unwrap();
    let out_slice = out.cuda_slice();
    let reg_info = map.get("strided_copy").expect("func_name not found");
    let cfg = compute_kernel_launch_config(out.device(), reg_info, out.size());
    unsafe {
        kernel.launch(
            cfg,
            (
                out_slice,
                x,
                &out.cuda_shape()?,
                &out.cuda_strides()?,
                out.ndim(),
                out.size(),
            ),
        )
    }?;
    Ok(())
}

pub(crate) fn arg_reduce_strided_copy<T, const DEVICE_ID: usize>(
    x: &CudaSlice<T>,
    out: &mut _Tensor<T, Cuda, DEVICE_ID>,
    inner_size: usize,
) -> std::result::Result<(), TensorError>
where
    T: CommonBounds + CudaType + DeviceRepr,
{
    let include = get_include_1::<T>();
    let module_name = format!("arg_reduce_strided_copy_{}", T::CUDA_TYPE);
    let map = compile_kernel(
        &module_name,
        &format!(
            "
                {include}
                extern \"C\" __global__ void strided_copy({} *out, {} *inp, long long *shape, long long *strides, size_t inner_size, size_t ndim, size_t size)
                {{
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t stride = blockDim.x * gridDim.x;
                    while (idx < size)
                    {{
                        long long amount = idx;
                        long long offset = 0;
                        #pragma unroll
                        for (int j = ndim - 1; j >= 0; j--)
                        {{
                            offset += amount % shape[j] * strides[j];
                            amount /= shape[j];
                        }}
                        out[offset] = inp[idx * inner_size];
                        idx += stride;
                    }}
                }}",
            T::CUDA_TYPE,
            T::CUDA_TYPE,
        ),
        out.device(),
        &["strided_copy"],
    )?;
    let kernel = out.device().get_func(&module_name, "strided_copy").unwrap();
    let out_slice = out.cuda_slice();
    let reg_info = map.get("strided_copy").expect("func_name not found");
    let cfg = compute_kernel_launch_config(out.device(), reg_info, out.size());
    unsafe {
        kernel.launch(
            cfg,
            (
                out_slice,
                x,
                &out.cuda_shape()?,
                &out.cuda_strides()?,
                inner_size,
                out.ndim(),
                out.size(),
            ),
        )
    }?;
    Ok(())
}
