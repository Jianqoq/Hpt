use std::{collections::HashMap, panic::Location, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use regex::Regex;
use tensor_common::err_handler::TensorError;
use tensor_cudakernels::RegisterInfo;
use tensor_types::dtype::TypeCommon;

use crate::{cuda_compiled::CUDA_COMPILED, tensor_base::_Tensor, Cuda};

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn compile_kernel(
    module: &str,
    code: &str,
    device: Arc<CudaDevice>,
    kernels: &[&'static str],
) -> std::result::Result<Arc<HashMap<String, RegisterInfo>>, TensorError> {
    if let Ok(mut cache) = CUDA_COMPILED.lock() {
        if let Some(set) = cache.get_mut(&device.ordinal()) {
            if let Some(reg_info) = set.get(module) {
                Ok(reg_info.clone())
            } else {
                let cuda_path = if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                    cuda_path
                } else {
                    return Err(TensorError::EnvVarNotSet(
                        "CUDA_PATH",
                        "compile_kernel",
                        Location::caller(),
                    ));
                };
                let mut opts = CompileOptions::default();
                opts.include_paths.push(format!("{}/include", cuda_path));
                let ptx = compile_ptx_with_opts(code, opts).map_err(|_| {
                    TensorError::CudaKernelCompileError(
                        module.to_string(),
                        code.to_string(),
                        Location::caller(),
                    )
                })?;
                let reg_counts = count_registers(&ptx.to_src());
                device.load_ptx(ptx, module, kernels).map_err(|x| {
                    TensorError::CudaLoadPTXFailed(
                        module.to_string(),
                        code.to_string(),
                        Location::caller(),
                        x,
                    )
                })?;
                set.insert(module.to_string(), Arc::new(reg_counts));
                Ok(set.get(module).unwrap().clone())
            }
        } else {
            let mut map = HashMap::new();
            let cuda_path = std::env::var("CUDA_PATH").unwrap();
            let mut opts = CompileOptions::default();
            opts.include_paths.push(format!("{}/include", cuda_path));
            let ptx = compile_ptx_with_opts(code, opts).map_err(|_| {
                TensorError::CudaKernelCompileError(
                    module.to_string(),
                    code.to_string(),
                    Location::caller(),
                )
            })?;
            let reg_counts = count_registers(&ptx.to_src());
            device.load_ptx(ptx, module, kernels).map_err(|x| {
                TensorError::CudaLoadPTXFailed(
                    module.to_string(),
                    code.to_string(),
                    Location::caller(),
                    x,
                )
            })?;
            let reg_counts = Arc::new(reg_counts);
            map.insert(module.to_string(), reg_counts.clone());
            cache.insert(device.ordinal(), map);
            Ok(reg_counts)
        }
    } else {
        Err(TensorError::LockFailed("compile_kernel", Location::caller()))
    }
}

pub(crate) fn load_ptx_and_get_data(
    module: &str,
    func_name: &str,
    device: Arc<CudaDevice>,
    cap: usize,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
) -> std::result::Result<(CudaFunction, RegisterInfo), TensorError> {
    if let Some(func) = device.get_func(module, func_name) {
        if let Some(meta) = meta.get(&cap) {
            if let Some(reg_info) = meta.1.get(func_name) {
                Ok((func, *reg_info))
            } else {
                Err(TensorError::CudaKernelReginfoNotFound(
                    module.to_string(),
                    func_name.to_string(),
                    Location::caller(),
                ))
            }
        } else {
            Err(TensorError::CudaKernelMetaNotFound(
                cap,
                module.to_string(),
                func_name.to_string(),
                Location::caller(),
            ))
        }
    } else {
        if let Some(meta) = meta.get(&cap) {
            device
                .load_ptx(meta.0.into(), module, meta.2)
                .map_err(|x| {
                    TensorError::CudaLoadPTXFailed(
                        module.to_string(),
                        meta.0.to_string(),
                        Location::caller(),
                        x,
                    )
                })?;
            if let Some(reg_info) = meta.1.get(func_name) {
                Ok((device.get_func(module, func_name).unwrap(), *reg_info))
            } else {
                Err(TensorError::CudaKernelReginfoNotFound(
                    module.to_string(),
                    func_name.to_string(),
                    Location::caller(),
                ))
            }
        } else {
            Err(TensorError::CudaKernelMetaNotFound(
                cap,
                module.to_string(),
                func_name.to_string(),
                Location::caller(),
            ))
        }
    }
}

fn count_registers(ptx: &str) -> HashMap<String, RegisterInfo> {
    let mut reg_counts = HashMap::new();

    let func_re = Regex::new(r"\.visible \.entry (\w+)\(").unwrap();
    let pred_re = Regex::new(r"\.reg \.pred\s+%p<(\d+)>").unwrap();
    let b32_re = Regex::new(r"\.reg \.b32\s+%r<(\d+)>").unwrap();
    let b64_re = Regex::new(r"\.reg \.b64\s+%rd<(\d+)>").unwrap();
    let b16_re = Regex::new(r"\.reg \.b16\s+%rs<(\d+)>").unwrap();
    let mut current_func = String::new();

    for line in ptx.lines() {
        if let Some(cap) = func_re.captures(line) {
            current_func = cap[1].to_string();
        } else if !current_func.is_empty() {
            if let Some(cap) = pred_re.captures(line) {
                let pred_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert(RegisterInfo {
                        pred: 0,
                        b32: 0,
                        b64: 0,
                        b16: 0,
                    })
                    .pred = pred_count;
            }
            if let Some(cap) = b32_re.captures(line) {
                let b32_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert(RegisterInfo {
                        pred: 0,
                        b32: 0,
                        b64: 0,
                        b16: 0,
                    })
                    .b32 = b32_count;
            }
            if let Some(cap) = b64_re.captures(line) {
                let b64_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert(RegisterInfo {
                        pred: 0,
                        b32: 0,
                        b64: 0,
                        b16: 0,
                    })
                    .b64 = b64_count;
            }
            if let Some(cap) = b16_re.captures(line) {
                let b16_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert(RegisterInfo {
                        pred: 0,
                        b32: 0,
                        b64: 0,
                        b16: 0,
                    })
                    .b16 = b16_count;
            }
        }
    }

    reg_counts
}

pub(crate) fn align_to_warp(threads: usize, warp_size: usize) -> usize {
    threads / warp_size * warp_size
}

pub(crate) fn compute_kernel_launch_config(
    device: Arc<CudaDevice>,
    reg_info: &RegisterInfo,
    size: usize,
) -> LaunchConfig {
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE;

    let max_regs_per_sm = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
        .expect("failed to get max registers per sm");
    let warp_size = device
        .attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        .expect("failed to get warp size");
    let max_threads_per_block = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        .expect("failed to get max threads per block");
    let total_reg_used = reg_info.b32 + reg_info.b64 * 2;
    let max_threads_per_sm = max_regs_per_sm as usize / total_reg_used;
    let aligned_threads = align_to_warp(max_threads_per_sm, warp_size as usize)
        .min(max_threads_per_block as usize)
        .max(warp_size as usize);
    LaunchConfig {
        block_dim: (aligned_threads as u32, 1, 1),
        grid_dim: (
            (size as u32 + aligned_threads as u32 - 1) / aligned_threads as u32,
            1,
            1,
        ),
        shared_mem_bytes: 0,
    }
}

pub(crate) fn get_include_1<T: TypeCommon>() -> &'static str {
    if T::CUDA_TYPE == "half" {
        "#include <cuda_fp16.h>"
    } else {
        ""
    }
}

pub(crate) fn get_module_name_1<T: TypeCommon, const CUDA_DEVICE: usize>(
    header: &str,
    tensor: &_Tensor<T, Cuda, CUDA_DEVICE>,
) -> String {
    format!(
        "{header}{}{}{}",
        T::ID,
        tensor.layout.shape(),
        tensor.layout.strides(),
    )
}

pub(crate) fn get_module_name_vec<T: TypeCommon, const CUDA_DEVICE: usize>(
    header: &str,
    tensors: &[_Tensor<T, Cuda, CUDA_DEVICE>],
) -> String {
    let mut string = String::new();
    for i in tensors.iter() {
        string.push_str(&format!(
            "{}{}{}",
            T::ID,
            i.layout.shape(),
            i.layout.strides(),
        ));
    }
    format!("{header}{}", string)
}

pub(crate) fn get_array_str(array: &[i64]) -> String {
    array
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}
