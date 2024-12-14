use std::{collections::HashMap, sync::Arc};

use cudarc::{
    driver::{CudaDevice, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use regex::Regex;

use crate::cuda_compiled::CUDA_COMPILED;

pub(crate) fn compile_kernel(
    module: &str,
    code: &str,
    device: Arc<CudaDevice>,
    kernels: &[&'static str],
) -> anyhow::Result<Arc<HashMap<String, RegisterInfo>>> {
    if let Ok(mut cache) = CUDA_COMPILED.lock() {
        if let Some(set) = cache.get_mut(&device.ordinal()) {
            if let Some(reg_info) = set.get(module) {
                Ok(reg_info.clone())
            } else {
                let cuda_path = std::env::var("CUDA_PATH").unwrap();
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
        Err(anyhow::anyhow!("failed to lock CUDA_COMPILED"))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegisterInfo {
    pub pred: usize,
    pub b32: usize,
    pub b64: usize,
}

fn count_registers(ptx: &str) -> HashMap<String, RegisterInfo> {
    let mut reg_counts = HashMap::new();

    // 匹配函数名和寄存器声明
    let func_re = Regex::new(r"\.visible \.entry (\w+)\(").unwrap();
    let pred_re = Regex::new(r"\.reg \.pred\s+%p<(\d+)>").unwrap();
    let b32_re = Regex::new(r"\.reg \.b32\s+%r<(\d+)>").unwrap();
    let b64_re = Regex::new(r"\.reg \.b64\s+%rd<(\d+)>").unwrap();

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
                    })
                    .b64 = b64_count;
            }
        }
    }

    reg_counts
}

pub(crate) fn align_to_warp(threads: usize, warp_size: usize) -> usize {
    threads & !(warp_size - 1)
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
    let aligned_threads =
        align_to_warp(max_threads_per_sm, warp_size as usize).min(max_threads_per_block as usize);
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
