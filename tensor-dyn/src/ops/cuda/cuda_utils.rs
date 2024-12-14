use std::{collections::HashSet, sync::Arc};

use cudarc::{driver::CudaDevice, nvrtc::{compile_ptx_with_opts, CompileOptions}};

use crate::cuda_compiled::CUDA_COMPILED;

pub(crate) fn compile_kernel(
    module: &str,
    code: &str,
    device: Arc<CudaDevice>,
    kernels: &[&'static str],
) -> anyhow::Result<()> {
    if let Ok(mut cache) = CUDA_COMPILED.lock() {
        if let Some(set) = cache.get_mut(&device.ordinal()) {
            if set.get(module).is_some() {
                Ok(())
            } else {
                let cuda_path = std::env::var("CUDA_PATH").unwrap();
                let mut opts = CompileOptions::default();
                opts.include_paths.push(format!("{}/include", cuda_path));
                let ptx = compile_ptx_with_opts(code, opts)?;
                device.load_ptx(ptx, module, kernels)?;
                set.insert(module.to_string());
                Ok(())
            }
        } else {
            let mut set = HashSet::new();
            let cuda_path = std::env::var("CUDA_PATH").unwrap();
            let mut opts = CompileOptions::default();
            opts.include_paths.push(format!("{}/include", cuda_path));
            let ptx = compile_ptx_with_opts(code, opts)?;
            device.load_ptx(ptx, module, kernels)?;
            set.insert(module.to_string());
            cache.insert(device.ordinal(), set);
            Ok(())
        }
    } else {
        Err(anyhow::anyhow!("failed to lock CUDA_COMPILED"))
    }
}
