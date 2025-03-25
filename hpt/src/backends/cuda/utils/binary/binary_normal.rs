use crate::backends::cuda::cuda_utils::compile_kernel;
use crate::backends::cuda::cuda_utils::compute_kernel_launch_config;
use crate::backends::cuda::cuda_utils::get_fast_divmod;
use crate::backends::cuda::cuda_utils::get_slice_i32;
use crate::backends::cuda::cuda_utils::load_ptx_and_get_data;
use crate::tensor_base::_Tensor;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchAsync;
use hpt_allocator::Cuda;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_cudakernels::RegisterInfo;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_traits::tensor::TensorLike;
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use std::borrow::BorrowMut;
use std::collections::HashSet;

use crate::backends::cuda::cuda_utils::get_array_str;
use crate::backends::cuda::cuda_utils::get_include_1;

use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};

#[track_caller]
pub(crate) fn binary_fn_with_out_simd<A, B, O, K, F, const CUDA_DEVICE: usize, Al>(
    op_name: &str,
    lhs: &_Tensor<A, Cuda, CUDA_DEVICE, Al>,
    rhs: &_Tensor<B, Cuda, CUDA_DEVICE, Al>,
    f: F,
    out: Option<O>,
) -> Result<_Tensor<K, Cuda, CUDA_DEVICE, Al>, TensorError>
where
    A: CommonBounds + DeviceRepr + CudaType,
    B: CommonBounds + DeviceRepr + CudaType,
    O: BorrowMut<_Tensor<K, Cuda, CUDA_DEVICE, Al>>,
    K: CommonBounds + DeviceRepr + CudaType,
    F: Fn(Scalar<K>, Scalar<A>, Scalar<B>) -> Scalar<K>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let module_name = format!(
        "bn{}{}{}{}{}{}{}",
        op_name,
        A::STR,
        lhs.layout.shape(),
        lhs.layout.strides(),
        B::STR,
        rhs.layout.shape(),
        rhs.layout.strides()
    );
    let k_include = get_include_1::<K>();
    let a_include = get_include_1::<A>();
    let b_include = get_include_1::<B>();
    let mut set = HashSet::new();
    set.insert(k_include);
    set.insert(a_include);
    set.insert(b_include);
    let includes = set.into_iter().collect::<Vec<_>>().join("\n");
    if lhs.size() == 1 {
        let val = lhs.to_cpu::<0>()?.as_raw()[0];
        let res = extract_out::<B, K, O, CUDA_DEVICE, Al>(rhs.shape(), out)?;
        if rhs.is_contiguous() {
            let out_scalar = Scalar::new("out[idx]".to_string());
            let lhs_scalar = Scalar::new("lhs".to_string());
            let rhs_scalar = Scalar::new("rhs[idx]".to_string());
            let map = compile_kernel(
                &module_name,
                &format!(
                    "
                    {includes}
                    extern \"C\" __global__ void lhs_scalar_rhs_contiguous({} *out, {} lhs, {} *rhs)
                    {{
                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        size_t stride = blockDim.x * gridDim.x;
                        while (idx < {})
                        {{
                            out[idx] = {};
                            idx += stride;
                        }}
                    }}",
                    K::CUDA_TYPE,
                    A::CUDA_TYPE,
                    B::CUDA_TYPE,
                    res.size(),
                    f(out_scalar, lhs_scalar, rhs_scalar).val()
                ),
                res.device(),
                &["lhs_scalar_rhs_contiguous"],
            )?;
            let kernel = res
                .device()
                .get_func(&module_name, "lhs_scalar_rhs_contiguous")
                .unwrap();
            let out_slice = res.cuda_slice();
            let rhs_slice = rhs.cuda_slice();
            let reg_info = map
                .get("lhs_scalar_rhs_contiguous")
                .expect("func_name not found");
            let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
            unsafe { kernel.launch(cfg, (out_slice, val, rhs_slice)) }?;
        } else {
            let rhs_broadcast_layout = rhs.layout.to_broadcast_layout(res.shape())?;
            let shape_str = get_array_str(rhs_broadcast_layout.shape());
            let strides_str = get_array_str(rhs_broadcast_layout.strides());
            let out_scalar = Scalar::new("out[idx]".to_string());
            let lhs_scalar = Scalar::new("lhs".to_string());
            let rhs_scalar = Scalar::new("rhs[offset]".to_string());
            let map = compile_kernel(
                &module_name,
                &format!(
                    "
                    {includes}
                    __constant__ long long shape[] = {{{}}};
                    __constant__ long long strides[] = {{{}}};
                    extern \"C\" __global__ void lhs_scalar_rhs_not_contiguous({} *out, {} lhs, {} *rhs)
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
                            {};
                            idx += stride;
                        }}
                    }}",
                    shape_str,
                    strides_str,
                    K::CUDA_TYPE,
                    A::CUDA_TYPE,
                    B::CUDA_TYPE,
                    res.size(),
                    res.ndim(),
                    f(out_scalar, lhs_scalar, rhs_scalar).val()
                ),
                res.device(),
                &["lhs_scalar_rhs_not_contiguous"],
            )?;
            let kernel = res
                .device()
                .get_func(&module_name, "lhs_scalar_rhs_not_contiguous")
                .unwrap();
            let out_slice = res.cuda_slice();
            let rhs_slice = rhs.cuda_slice();
            let reg_info = map
                .get("lhs_scalar_rhs_not_contiguous")
                .expect("func_name not found");
            let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
            unsafe { kernel.launch(cfg, (out_slice, val, rhs_slice)) }?;
        }
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.to_cpu::<0>()?.as_raw()[0];
        let res = extract_out::<A, K, O, CUDA_DEVICE, Al>(lhs.shape(), out)?;
        if lhs.is_contiguous() {
            let out_scalar = Scalar::new("out[idx]".to_string());
            let lhs_scalar = Scalar::new("lhs[idx]".to_string());
            let rhs_scalar = Scalar::new("rhs".to_string());
            let map = compile_kernel(
                &module_name,
                &format!(
                    "
                    {includes}
                    extern \"C\" __global__ void rhs_scalar_lhs_contiguous({} *out, {} *lhs, {} rhs)
                    {{
                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        size_t stride = blockDim.x * gridDim.x;
                        while (idx < {})
                        {{
                            {};
                            idx += stride;
                        }}
                    }}",
                    K::CUDA_TYPE,
                    A::CUDA_TYPE,
                    B::CUDA_TYPE,
                    res.size(),
                    f(out_scalar, lhs_scalar, rhs_scalar).val()
                ),
                res.device(),
                &["rhs_scalar_lhs_contiguous"],
            )?;
            let kernel = res
                .device()
                .get_func(&module_name, "rhs_scalar_lhs_contiguous")
                .unwrap();
            let out_slice = res.cuda_slice();
            let lhs_slice = lhs.cuda_slice();
            let reg_info = map
                .get("rhs_scalar_lhs_contiguous")
                .expect("func_name not found");
            let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
            unsafe { kernel.launch(cfg, (out_slice, lhs_slice, val)) }?;
        } else {
            let lhs_broadcast_layout = lhs.layout.to_broadcast_layout(res.shape())?;
            let shape_str = get_array_str(lhs_broadcast_layout.shape());
            let strides_str = get_array_str(lhs_broadcast_layout.strides());
            let out_scalar = Scalar::new("out[idx]".to_string());
            let lhs_scalar = Scalar::new("lhs[offset]".to_string());
            let rhs_scalar = Scalar::new("rhs".to_string());
            let map = compile_kernel(
                &module_name,
                &format!(
                    "
                    {includes}
                    __constant__ long long shape[] = {{{shape_str}}};
                    __constant__ long long strides[] = {{{strides_str}}};
                    extern \"C\" __global__ void rhs_scalar_lhs_not_contiguous({} *out, {} *lhs, {} rhs)
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
                            {};
                            idx += stride;
                        }}
                    }}",
                    K::CUDA_TYPE,
                    A::CUDA_TYPE,
                    B::CUDA_TYPE,
                    res.size(),
                    res.ndim(),
                    f(out_scalar, lhs_scalar, rhs_scalar).val()
                ),
                res.device(),
                &["rhs_scalar_lhs_not_contiguous"],
            )?;
            let kernel = res
                .device()
                .get_func(&module_name, "rhs_scalar_lhs_not_contiguous")
                .unwrap();
            let out_slice = res.cuda_slice();
            let lhs_slice = lhs.cuda_slice();
            let reg_info = map
                .get("rhs_scalar_lhs_not_contiguous")
                .expect("func_name not found");
            let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
            unsafe { kernel.launch(cfg, (out_slice, lhs_slice, val)) }?;
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let res = extract_out::<B, K, O, CUDA_DEVICE, Al>(rhs.shape(), out)?;
            let out_scalar = Scalar::new("out[idx]".to_string());
            let lhs_scalar = Scalar::new("lhs[idx]".to_string());
            let rhs_scalar = Scalar::new("rhs[idx]".to_string());
            let map = compile_kernel(
                &module_name,
                &format!(
                    "
                    {includes}
                    extern \"C\" __global__ void lhs_scalar_rhs_contiguous({} *out, {} *lhs, {} *rhs)
                    {{
                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        size_t stride = blockDim.x * gridDim.x;
                        while (idx < {})
                        {{
                            {};
                            idx += stride;
                        }}
                    }}",
                    K::CUDA_TYPE,
                    A::CUDA_TYPE,
                    B::CUDA_TYPE,
                    res.size(),
                    f(out_scalar, lhs_scalar, rhs_scalar).val()
                ),
                res.device(),
                &["lhs_scalar_rhs_contiguous"],
            )?;
            let kernel = res
                .device()
                .get_func(&module_name, "lhs_scalar_rhs_contiguous")
                .unwrap();
            let out_slice = res.cuda_slice();
            let rhs_slice = rhs.cuda_slice();
            let lhs_slice = lhs.cuda_slice();
            let reg_info = map
                .get("lhs_scalar_rhs_contiguous")
                .expect("func_name not found");
            let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
            unsafe { kernel.launch(cfg, (out_slice, lhs_slice, rhs_slice)) }?;
            Ok(res)
        } else {
            let res_layout = lhs.layout.broadcast(&rhs.layout)?;
            let lhs_broadcast_layout = lhs.layout.to_broadcast_layout(res_layout.shape())?;
            let rhs_broadcast_layout = rhs.layout.to_broadcast_layout(res_layout.shape())?;
            let res = extract_out::<K, K, O, CUDA_DEVICE, Al>(res_layout.shape(), out)?;
            let lhs_shape_str = get_array_str(lhs_broadcast_layout.shape());
            let lhs_strides_str = get_array_str(lhs_broadcast_layout.strides());
            let rhs_shape_str = get_array_str(rhs_broadcast_layout.shape());
            let rhs_strides_str = get_array_str(rhs_broadcast_layout.strides());
            let out_scalar = Scalar::new("out[idx]".to_string());
            let lhs_scalar = Scalar::new("lhs[lhs_offset]".to_string());
            let rhs_scalar = Scalar::new("rhs[rhs_offset]".to_string());
            let map = compile_kernel(
                &module_name,
                &format!(
                    "
                    {includes}
                    __constant__ long long lhs_shape[] = {{{lhs_shape_str}}};
                    __constant__ long long lhs_strides[] = {{{lhs_strides_str}}};
                    __constant__ long long rhs_shape[] = {{{rhs_shape_str}}};
                    __constant__ long long rhs_strides[] = {{{rhs_strides_str}}};
                    extern \"C\" __global__ void binop({} *out, {} *lhs, {} *rhs)
                    {{
                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        size_t stride = blockDim.x * gridDim.x;
                        while (idx < {})
                        {{
                            long lhs_amount = idx;
                            long lhs_offset = 0;
                            long rhs_amount = idx;
                            long rhs_offset = 0;
                            #pragma unroll
                            for (int j = {} - 1; j >= 0; j--)
                            {{
                                lhs_offset += lhs_amount % lhs_shape[j] * lhs_strides[j];
                                lhs_amount /= lhs_shape[j];
                                rhs_offset += rhs_amount % rhs_shape[j] * rhs_strides[j];
                                rhs_amount /= rhs_shape[j];
                            }}
                            {};
                            idx += stride;
                        }}
                    }}",
                    K::CUDA_TYPE,
                    A::CUDA_TYPE,
                    B::CUDA_TYPE,
                    res.size(),
                    res.ndim(),
                    f(out_scalar, lhs_scalar, rhs_scalar).val()
                ),
                res.device(),
                &["binop"],
            )?;
            let kernel = res
                .device()
                .get_func(&module_name, "binop")
                .expect("func_name not found");
            let out_slice = res.cuda_slice();
            let rhs_slice = rhs.cuda_slice();
            let lhs_slice = lhs.cuda_slice();
            let reg_info = map.get("binop").expect("func_name not found");
            let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
            unsafe { kernel.launch(cfg, (out_slice, lhs_slice, rhs_slice)) }?;
            Ok(res)
        }
    }
}

#[track_caller]
pub(crate) fn binary_fn_precompiled<A, B, O, K, const CUDA_DEVICE: usize, Al>(
    lhs: &_Tensor<A, Cuda, CUDA_DEVICE, Al>,
    rhs: &_Tensor<B, Cuda, CUDA_DEVICE, Al>,
    op_name: &str,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
    out: Option<O>,
) -> Result<_Tensor<K, Cuda, CUDA_DEVICE, Al>, TensorError>
where
    A: CommonBounds + DeviceRepr + CudaType,
    B: CommonBounds + DeviceRepr + CudaType,
    O: BorrowMut<_Tensor<K, Cuda, CUDA_DEVICE, Al>>,
    K: CommonBounds + DeviceRepr + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    if lhs.size() == 1 {
        let val = lhs.to_cpu::<0>()?.as_raw()[0];
        let res = extract_out::<B, K, O, CUDA_DEVICE, Al>(rhs.shape(), out)?;
        if rhs.is_contiguous() {
            let (kernel, reg_info) = load_ptx_and_get_data(
                op_name,
                &format!("{op_name}_{}_{}_contiguous_lhs_scalar", A::STR, B::STR),
                res.device(),
                res.device_cap(),
                meta,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
            unsafe {
                kernel.launch(
                    cfg,
                    (res.cuda_slice(), val, rhs.cuda_slice(), res.size() as i32),
                )
            }?;
        } else {
            let (kernel, reg_info) = load_ptx_and_get_data(
                op_name,
                &format!("{op_name}_{}_{}_uncontiguous_lhs_scalar", A::STR, B::STR),
                res.device(),
                res.device_cap(),
                meta,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
            let rhs_fast_divmod = rhs.cuda_divmod()?;
            let rhs_strides = rhs.cuda_strides_i32()?;
            unsafe {
                kernel.launch(
                    cfg,
                    (
                        res.cuda_slice(),
                        val,
                        rhs.cuda_slice(),
                        &rhs_fast_divmod,
                        &rhs_strides,
                        rhs.ndim() as i32,
                        res.size() as i32,
                    ),
                )
            }?;
        }
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.to_cpu::<0>()?.as_raw()[0];
        let res = extract_out::<A, K, O, CUDA_DEVICE, Al>(lhs.shape(), out)?;
        if lhs.is_contiguous() {
            let (kernel, reg_info) = load_ptx_and_get_data(
                op_name,
                &format!("{op_name}_{}_{}_contiguous_rhs_scalar", A::STR, B::STR),
                res.device(),
                res.device_cap(),
                meta,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
            unsafe {
                kernel.launch(
                    cfg,
                    (res.cuda_slice(), lhs.cuda_slice(), val, res.size() as i32),
                )
            }?;
        } else {
            let (kernel, reg_info) = load_ptx_and_get_data(
                op_name,
                &format!("{op_name}_{}_{}_uncontiguous_rhs_scalar", A::STR, B::STR),
                res.device(),
                res.device_cap(),
                meta,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
            let lhs_fast_divmod = lhs.cuda_divmod()?;
            let lhs_strides = lhs.cuda_strides_i32()?;
            unsafe {
                kernel.launch(
                    cfg,
                    (
                        res.cuda_slice(),
                        lhs.cuda_slice(),
                        val,
                        &lhs_fast_divmod,
                        &lhs_strides,
                        lhs.ndim() as i32,
                        res.size() as i32,
                    ),
                )
            }?;
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let res = extract_out::<B, K, O, CUDA_DEVICE, Al>(rhs.shape(), out)?;
            let (kernel, reg_info) = load_ptx_and_get_data(
                op_name,
                &format!("{op_name}_{}_{}_contiguous", A::STR, B::STR),
                res.device(),
                res.device_cap(),
                meta,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
            unsafe {
                kernel.launch(
                    cfg,
                    (
                        res.cuda_slice(),
                        lhs.cuda_slice(),
                        rhs.cuda_slice(),
                        res.size() as i32,
                    ),
                )
            }?;
            Ok(res)
        } else {
            let res_layout = lhs.layout.broadcast(&rhs.layout)?;
            let lhs_broadcast_layout = lhs.layout.to_broadcast_layout(res_layout.shape())?;
            let rhs_broadcast_layout = rhs.layout.to_broadcast_layout(res_layout.shape())?;
            let res = extract_out::<K, K, O, CUDA_DEVICE, Al>(res_layout.shape(), out)?;
            let (kernel, reg_info) = load_ptx_and_get_data(
                op_name,
                &format!("{op_name}_{}_{}_uncontiguous", A::STR, B::STR),
                res.device(),
                res.device_cap(),
                meta,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());

            let lhs_fast_divmod = get_fast_divmod(lhs_broadcast_layout.shape(), res.device())?;
            let lhs_strides = get_slice_i32(lhs_broadcast_layout.strides(), res.device())?;
            let rhs_fast_divmod = get_fast_divmod(rhs_broadcast_layout.shape(), res.device())?;
            let rhs_strides = get_slice_i32(rhs_broadcast_layout.strides(), res.device())?;
            unsafe {
                kernel.launch(
                    cfg,
                    (
                        res.cuda_slice(),
                        lhs.cuda_slice(),
                        &lhs_fast_divmod,
                        &lhs_strides,
                        rhs.cuda_slice(),
                        &rhs_fast_divmod,
                        &rhs_strides,
                        lhs_broadcast_layout.ndim() as i32,
                        rhs_broadcast_layout.ndim() as i32,
                        res.size() as i32,
                    ),
                )
            }?;
            Ok(res)
        }
    }
}

fn extract_out<A, K, O, const CUDA_DEVICE: usize, Al>(
    res_shape: &Shape,
    out: Option<O>,
) -> Result<_Tensor<K, Cuda, CUDA_DEVICE, Al>, TensorError>
where
    A: CommonBounds + DeviceRepr,
    K: CommonBounds + DeviceRepr + CudaType,
    O: BorrowMut<_Tensor<K, Cuda, CUDA_DEVICE, Al>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let ret = if let Some(mut out) = out {
        ShapeError::check_inplace_out_layout_valid(res_shape, &out.borrow().layout())?;
        (*out.borrow_mut()).clone()
    } else {
        _Tensor::<K, Cuda, CUDA_DEVICE, Al>::empty(res_shape)?
    };
    Ok(ret)
}
