use crate::ops::cuda::cuda_utils::{compile_kernel, compute_kernel_launch_config, get_array_str};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::{DeviceRepr, LaunchAsync};
use std::borrow::Borrow;
use std::panic::Location;
use tensor_common::err_handler::ErrHandler::{self, InvalidOutSize};
use tensor_traits::tensor::CommonBounds;
use tensor_traits::{TensorCreator, TensorInfo};
use tensor_types::cuda_types::scalar::Scalar;

pub(crate) fn uary_fn_with_out_simd<A, O, K, F, const DEVICE_ID: usize>(
    inp: &_Tensor<A, Cuda, DEVICE_ID>,
    module_name: &str,
    f: F,
    out: Option<O>,
) -> std::result::Result<_Tensor<K, Cuda, DEVICE_ID>, ErrHandler>
where
    A: CommonBounds + DeviceRepr,
    K: CommonBounds + DeviceRepr,
    O: Borrow<_Tensor<K, Cuda, DEVICE_ID>>,
    F: Fn(Scalar<K>, Scalar<A>) -> Scalar<K>,
{
    let ret = if let Some(out) = out {
        if out.borrow().size() * size_of::<K>() == inp.size() * size_of::<A>() {
            out.borrow().static_cast()?
        } else {
            return Err(InvalidOutSize(
                inp.size() * size_of::<A>(),
                out.borrow().size() * size_of::<K>(),
                Location::caller(),
            )
            .into());
        }
    } else {
        _Tensor::<K, Cuda, DEVICE_ID>::empty(inp.shape())?
    };
    let half_include = if K::CUDA_TYPE == "half" || A::CUDA_TYPE == "half" {
        "#include <cuda_fp16.h>"
    } else {
        ""
    };
    let bfloat16_include = if K::CUDA_TYPE == "bfloat16" || A::CUDA_TYPE == "bfloat16" {
        "#include <cuda_bf16.h>"
    } else {
        ""
    };
    let code = if inp.is_contiguous() || inp.parent().is_some() {
        let scalar_a = Scalar::<A>::new("inp[idx]".to_string());
        let scalar_k = Scalar::<K>::new("out[idx]".to_string());
        format!(
            "
        {half_include}
        {bfloat16_include}
        extern \"C\" __global__ void unary(const {}* inp, {}* out, int size) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {{
                {};
            }}
        }}
        ",
            A::CUDA_TYPE,
            K::CUDA_TYPE,
            f(scalar_k, scalar_a).val()
        )
    } else {
        let shape_str = get_array_str(inp.shape());
        let strides_str = get_array_str(inp.strides());
        let scalar_a = Scalar::<A>::new("inp[idx]".to_string());
        let scalar_k = Scalar::<K>::new("out[idx]".to_string());
        format!(
            "
        {half_include}
        {bfloat16_include}
        __constant__ long long shape[] = {{{}}};
        __constant__ long long strides[] = {{{}}};
        extern \"C\" __global__ void unary(const {}* inp, {}* out, int size) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {{
                long amount = idx;
                long offset = 0;
                #pragma unroll
                for (int j = {} - 1; j >= 0; j--)
                {{
                    offset += amount % shape[j] * strides[j];
                    amount /= shape[j];
                }}
                {};
            }}
        }}
        ",
            shape_str,
            strides_str,
            A::CUDA_TYPE,
            K::CUDA_TYPE,
            inp.shape().len(),
            f(scalar_k, scalar_a).val()
        )
    };
    let map = compile_kernel(module_name, &code, inp.device(), &["unary"])?;
    let kernel = ret.device().get_func(module_name, "unary").unwrap();
    let out_slice = ret.cuda_slice();
    let inp_slice = inp.cuda_slice();
    let reg_info = map.get("unary").expect("func_name not found");
    let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
    unsafe { kernel.launch(cfg, (inp_slice, out_slice, ret.size())) }.map_err(|e| {
        ErrHandler::CudaKernelLaunchingError(
            module_name.to_string(),
            "unary".to_string(),
            Location::caller(),
            e,
        )
    })?;
    Ok(ret)
}
