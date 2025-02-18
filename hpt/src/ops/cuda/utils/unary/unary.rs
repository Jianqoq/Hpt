use crate::ops::cuda::cuda_utils::{compile_kernel, compute_kernel_launch_config, get_array_str};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync};
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::{TensorCreator, TensorInfo};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use std::borrow::BorrowMut;

use crate::ops::cuda::cuda_utils::get_include_1;

pub(crate) fn uary_fn_with_out_simd<A, O, K, F, const DEVICE_ID: usize>(
    inp: &_Tensor<A, Cuda, DEVICE_ID>,
    module_name: &str,
    f: F,
    out: Option<O>,
) -> std::result::Result<_Tensor<K, Cuda, DEVICE_ID>, TensorError>
where
    A: CommonBounds + DeviceRepr + CudaType,
    K: CommonBounds + DeviceRepr + CudaType,
    O: BorrowMut<_Tensor<K, Cuda, DEVICE_ID>>,
    F: Fn(Scalar<K>, Scalar<A>) -> Scalar<K>,
{
    let ret = if let Some(mut out) = out {
        ShapeError::check_size_match(inp.size() as i64, out.borrow_mut().size() as i64)?;
        (*out.borrow_mut()).clone()
    } else {
        _Tensor::<K, Cuda, DEVICE_ID>::empty(inp.shape())?
    };
    let a_include = get_include_1::<A>();
    let k_include = get_include_1::<K>();
    let code = if inp.is_contiguous() && !inp.parent().is_some() {
        let scalar_a = Scalar::<A>::new("inp[idx]".to_string());
        let scalar_k = Scalar::<K>::new("out[idx]".to_string());
        format!(
            "
        {a_include}
        {k_include}
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
        let scalar_a = Scalar::<A>::new("inp[offset]".to_string());
        let scalar_k = Scalar::<K>::new("out[idx]".to_string());
        format!(
            "
        {a_include}
        {k_include}
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
    unsafe { kernel.launch(cfg, (inp_slice, out_slice, ret.size())) }?;
    Ok(ret)
}

pub(crate) fn unary_raw_mut<T, F>(
    slice: &CudaSlice<T>,
    res_slice: &CudaSlice<T>,
    module_name: &str,
    f: F,
) -> std::result::Result<(), TensorError>
where
    T: CudaType + CommonBounds + DeviceRepr,
    F: Fn(Scalar<T>, Scalar<T>) -> Scalar<T>,
{
    assert_eq!(slice.device().ordinal(), res_slice.device().ordinal());
    let a_include = get_include_1::<T>();
    let scalar_a = Scalar::<T>::new("inp[idx]".to_string());
    let scalar_k = Scalar::<T>::new("out[idx]".to_string());
    let code = format!(
        "
        {a_include}
        extern \"C\" __global__ void unary_raw(const {}* inp, {}* out, int size) {{
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {{
                {};
            }}
        }}
        ",
        T::CUDA_TYPE,
        T::CUDA_TYPE,
        f(scalar_k, scalar_a).val()
    );
    let map = compile_kernel(module_name, &code, slice.device(), &["unary_raw"])?;
    let kernel = slice.device().get_func(module_name, "unary_raw").unwrap();
    let reg_info = map.get("unary_raw").expect("func_name not found");
    let cfg = compute_kernel_launch_config(slice.device(), reg_info, slice.len());
    unsafe { kernel.launch(cfg, (slice, res_slice, slice.len())) }?;
    Ok(())
}
