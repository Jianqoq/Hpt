use std::{collections::HashMap, panic::Location, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use hpt_allocator::traits::Allocator;
use hpt_common::error::{
    base::TensorError, common::CommonError, device::DeviceError, kernel::KernelError,
};
use hpt_cudakernels::RegisterInfo;
use hpt_types::dtype::{CudaType, TypeCommon};
use regex::Regex;

use crate::{
    backend::Cuda, backends::common::divmod::FastDivmod, cuda_compiled::CUDA_COMPILED,
    tensor_base::_Tensor,
};

#[track_caller]
pub(crate) fn compile_kernel(
    module: &str,
    code: &str,
    device: Arc<CudaDevice>,
    kernels: &[&'static str],
) -> Result<Arc<HashMap<String, RegisterInfo>>, TensorError> {
    if let Ok(mut cache) = CUDA_COMPILED.lock() {
        if let Some(set) = cache.get_mut(&device.ordinal()) {
            if let Some(reg_info) = set.get(module) {
                Ok(reg_info.clone())
            } else {
                let cuda_path = if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                    cuda_path
                } else {
                    return Err(DeviceError::EnvVarNotSet {
                        variable: "CUDA_PATH".to_string(),
                        location: Location::caller(),
                    }
                    .into());
                };
                let mut opts = CompileOptions::default();
                opts.include_paths.push(format!("{}/include", cuda_path));
                let ptx = compile_ptx_with_opts(code, opts)?;
                let reg_counts = count_registers(&ptx.to_src());
                device.load_ptx(ptx, module, kernels)?;
                set.insert(module.to_string(), Arc::new(reg_counts));
                Ok(set.get(module).unwrap().clone())
            }
        } else {
            let mut map = HashMap::new();
            let cuda_path = std::env::var("CUDA_PATH").unwrap();
            let mut opts = CompileOptions::default();
            opts.include_paths.push(format!("{}/include", cuda_path));
            let ptx = compile_ptx_with_opts(code, opts)?;
            let reg_counts = count_registers(&ptx.to_src());
            device.load_ptx(ptx, module, kernels)?;
            let reg_counts = Arc::new(reg_counts);
            map.insert(module.to_string(), reg_counts.clone());
            cache.insert(device.ordinal(), map);
            Ok(reg_counts)
        }
    } else {
        Err(CommonError::LockFailed {
            message: "compile_kernel".to_string(),
            location: Location::caller(),
        }
        .into())
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
) -> Result<(CudaFunction, RegisterInfo), TensorError> {
    if let Some(func) = device.get_func(module, func_name) {
        if let Some(meta) = meta.get(&cap) {
            if let Some(reg_info) = meta.1.get(func_name) {
                Ok((func, *reg_info))
            } else {
                Err(KernelError::CudaKernelRegInfoNotFound {
                    module: module.to_string(),
                    func_name: func_name.to_string(),
                    location: Location::caller(),
                }
                .into())
            }
        } else {
            Err(KernelError::CudaKernelMetaNotFound {
                module: module.to_string(),
                func_name: func_name.to_string(),
                cap,
                location: Location::caller(),
            }
            .into())
        }
    } else {
        if let Some(meta) = meta.get(&cap) {
            device.load_ptx(meta.0.into(), module, meta.2)?;
            if let Some(reg_info) = meta.1.get(func_name) {
                Ok((device.get_func(module, func_name).unwrap(), *reg_info))
            } else {
                Err(KernelError::CudaKernelRegInfoNotFound {
                    module: module.to_string(),
                    func_name: func_name.to_string(),
                    location: Location::caller(),
                }
                .into())
            }
        } else {
            Err(KernelError::CudaKernelMetaNotFound {
                cap,
                module: module.to_string(),
                func_name: func_name.to_string(),
                location: Location::caller(),
            }
            .into())
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

pub(crate) fn compute_kernel_launch_config(
    device: Arc<CudaDevice>,
    _: &RegisterInfo,
    size: usize,
) -> LaunchConfig {
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    let sm_count = device
        .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("failed to get sm count") as usize;

    let mut block_size = if size > 1_000_000 { 256 } else { 128 };

    let mut grid_size = compute_num_blocks(device, size, block_size, 32);

    while grid_size < sm_count && block_size > 32 {
        block_size = (block_size / 2).max(32);

        grid_size = size.div_ceil(block_size);

        if block_size == 32 {
            break;
        }
    }

    LaunchConfig {
        block_dim: (block_size as u32, 1, 1),
        grid_dim: (grid_size as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(crate) fn compute_num_blocks(
    device: Arc<CudaDevice>,
    size: usize,
    block_size: usize,
    num_waves: usize,
) -> usize {
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    let sm_count = device
        .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("failed to get sm count");
    let tpm = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .expect("failed to get tpm");
    size.div_ceil(block_size)
        .min(sm_count as usize * tpm as usize / block_size * num_waves)
        .max(1)
}

pub(crate) fn check_launch_config(
    device: Arc<CudaDevice>,
    config: &LaunchConfig,
) -> Result<(), TensorError> {
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
    let max_grid_dim_x = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
        .expect("failed to get max grid dim");
    let max_grid_dim_y = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
        .expect("failed to get max grid dim");
    let max_grid_dim_z = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
        .expect("failed to get max grid dim");
    let max_block_size = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        .expect("failed to get max block dim");
    let max_shared_mem = device
        .attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        .expect("failed to get max shared mem");
    if config.grid_dim.0 > max_grid_dim_x as u32 {
        return Err(KernelError::LaunchConfigError {
            msg: format!(
                "max_grid_dim_x is {}, got {}",
                max_grid_dim_x, config.grid_dim.0
            ),
            location: Location::caller(),
        }
        .into());
    }
    if config.grid_dim.1 > max_grid_dim_y as u32 {
        return Err(KernelError::LaunchConfigError {
            msg: format!(
                "max_grid_dim_y is {}, got {}",
                max_grid_dim_y, config.grid_dim.1
            ),
            location: Location::caller(),
        }
        .into());
    }
    if config.grid_dim.2 > max_grid_dim_z as u32 {
        return Err(KernelError::LaunchConfigError {
            msg: format!(
                "max_grid_dim_z is {}, got {}",
                max_grid_dim_z, config.grid_dim.2
            ),
            location: Location::caller(),
        }
        .into());
    }
    if config.block_dim.0 * config.block_dim.1 * config.block_dim.2 > max_block_size as u32 {
        return Err(KernelError::LaunchConfigError {
            msg: format!(
                "max_block_size is {}, got {}",
                max_block_size,
                config.block_dim.0 * config.block_dim.1 * config.block_dim.2
            ),
            location: Location::caller(),
        }
        .into());
    }
    if config.shared_mem_bytes > max_shared_mem as u32 {
        return Err(KernelError::LaunchConfigError {
            msg: format!(
                "max_shared_mem is {}, got {}",
                max_shared_mem, config.shared_mem_bytes
            ),
            location: Location::caller(),
        }
        .into());
    }
    Ok(())
}

pub(crate) fn get_include_1<T: TypeCommon + CudaType>() -> &'static str {
    if T::CUDA_TYPE == "__half" {
        "#include <cuda_fp16.h>"
    } else if T::CUDA_TYPE == "__nv_bfloat16" {
        "#include <cuda_bf16.h>"
    } else {
        ""
    }
}

pub(crate) fn get_module_name_1<T: TypeCommon, const CUDA_DEVICE: usize, Al>(
    header: &str,
    tensor: &_Tensor<T, Cuda, CUDA_DEVICE, Al>,
) -> String
where
    Al: Allocator,
{
    format!(
        "{header}{}{}{}",
        T::STR,
        tensor.layout.shape(),
        tensor.layout.strides(),
    )
}

pub(crate) fn get_module_name_vec<T: TypeCommon, const CUDA_DEVICE: usize, Al>(
    header: &str,
    tensors: &[_Tensor<T, Cuda, CUDA_DEVICE, Al>],
) -> String
where
    Al: Allocator,
{
    let mut string = String::new();
    for i in tensors.iter() {
        string.push_str(&format!(
            "{}{}{}",
            T::STR,
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

pub(crate) fn get_fast_divmod(
    shape: &[i64],
    device: Arc<CudaDevice>,
) -> Result<CudaSlice<FastDivmod>, TensorError> {
    let mut res = vec![];
    for i in shape.iter() {
        let fast_divmod = FastDivmod::new(*i as i32);
        res.push(fast_divmod);
    }
    let res = device.htod_sync_copy(&res)?;
    Ok(res)
}

pub(crate) fn get_slice_i32(
    slice: &[i64],
    device: Arc<CudaDevice>,
) -> std::result::Result<cudarc::driver::CudaSlice<i32>, TensorError> {
    let slice = slice.iter().map(|x| *x as i32).collect::<Vec<_>>();
    let res = device.htod_sync_copy(&slice)?;
    Ok(res)
}
