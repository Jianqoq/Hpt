use ops::cpu::convolutions::conv2d_unroll::conv2d_ex_f32;
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

use winapi::um::psapi::{ GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS };
use winapi::um::processthreadsapi::GetCurrentProcess;
use std::mem::MaybeUninit;

fn _get_memory_info() -> Result<PROCESS_MEMORY_COUNTERS, String> {
    unsafe {
        // 获取当前进程句柄
        let process_handle = GetCurrentProcess();

        // 创建一个未初始化的结构体来存储内存信息
        let mut mem_counters = MaybeUninit::<PROCESS_MEMORY_COUNTERS>::uninit();

        // 调用 GetProcessMemoryInfo API
        if
            GetProcessMemoryInfo(
                process_handle,
                mem_counters.as_mut_ptr(),
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32
            ) == 0
        {
            return Err("Failed to get process memory information.".to_string());
        }

        // 将结构体变为初始化状态
        Ok(mem_counters.assume_init())
    }
}

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 8 * 80 * 4 * 4)?
        .reshape([80, 8, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 8 * 1260 * 1260)?
        .reshape([8, 1260, 1260])?
        .permute([1, 2, 0])?
        .contiguous()?;

    let now = std::time::Instant::now();
    for _ in 0..100 {
        let _ = conv2d_ex_f32(
            &a,
            &kernel,
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [2, 2]
        )?.permute([2, 0, 1])?;
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
