use crate::ops::cuda::cuda_utils::{
    compile_kernel, compute_kernel_launch_config, get_array_str, get_include_1, get_module_name_1,
};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use crate::TensorInfo;
use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync};
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
    let shape_str = get_array_str(out.shape());
    let strides_str = get_array_str(out.strides());
    let include = get_include_1::<T>();
    let module_name = get_module_name_1("strided_copy", &out);
    let map = compile_kernel(
        &module_name,
        &format!(
            "
                {include}
                __constant__ long long shape[] = {{{shape_str}}};
                __constant__ long long strides[] = {{{strides_str}}};
                extern \"C\" __global__ void strided_copy({} *out, {} *inp)
                {{
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t stride = blockDim.x * gridDim.x;
                    while (idx < {})
                    {{
                        long amount = idx;
                        long offset = 0;
                        #pragma unroll
                        for (int j = {} - 1; j >= 0; j--)
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
            x.len(),
            out.ndim(),
        ),
        out.device(),
        &["strided_copy"],
    )?;
    let kernel = out.device().get_func(&module_name, "strided_copy").unwrap();
    let out_slice = out.cuda_slice();
    let reg_info = map.get("strided_copy").expect("func_name not found");
    let cfg = compute_kernel_launch_config(out.device(), reg_info, out.size());
    unsafe { kernel.launch(cfg, (out_slice, x)) }?;
    Ok(())
}
