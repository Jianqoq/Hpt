use ops::cpu::convolutions::conv2d_unroll::{conv2d_ex_f32, conv2d_ex_i32};
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

fn conv2d_grouped(
    input: &Vec<Vec<Vec<f32>>>,
    kernels: &Vec<Vec<Vec<Vec<f32>>>>,
    groups: usize,
    stride: usize,
    padding: usize,
    dilation: usize
) -> Vec<Vec<Vec<f32>>> {
    let in_channels = input.len();
    let in_height = input[0].len();
    let in_width = input[0][0].len();
    let num_kernels = kernels.len();
    let kernel_height = kernels[0][0].len();
    let kernel_width = kernels[0][0][0].len();

    let in_channels_per_group: usize = in_channels / groups;
    let kernels_per_group: usize = num_kernels / groups;

    let output_height: usize =
        (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let output_width: usize =
        (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let mut output: Vec<Vec<Vec<f32>>> =
        vec![vec![vec![0.0; output_width]; output_height]; num_kernels];

    for g in 0..groups {
        for k in 0..kernels_per_group {
            for c in 0..in_channels_per_group {
                for y in 0..output_height {
                    // 根据 padding 和 stride，计算输入的有效范围
                    let in_y_start = (y as isize) * (stride as isize) - (padding as isize);
                    let in_y_end = in_y_start + (kernel_height as isize) * (dilation as isize);
                    // 限制有效的 y 范围，确保 input_y 在 [0, in_height) 之间
                    let y_start = (((0).max(in_y_start) as usize) / dilation).min(kernel_height);
                    let y_end = ((in_height.min(in_y_end as usize) as usize) / dilation).min(
                        kernel_height
                    );

                    for x in 0..output_width {
                        let in_x_start = (x as isize) * (stride as isize) - (padding as isize);
                        let in_x_end = in_x_start + (kernel_width as isize) * (dilation as isize);
                        // 限制有效的 x 范围，确保 input_x 在 [0, in_width) 之间
                        let x_start = (((0).max(in_x_start) as usize) / dilation).min(kernel_width);
                        let x_end = ((in_width.min(in_x_end as usize) as usize) / dilation).min(
                            kernel_width
                        );

                        let mut sum = 0.0;
                        for i in y_start..y_end {
                            let input_y = in_y_start + (i as isize) * (dilation as isize);
                            for j in x_start..x_end {
                                let input_x = in_x_start + (j as isize) * (dilation as isize);
                                // input_y 和 input_x 一定在合法范围内
                                let in_val =
                                    input[g * in_channels_per_group + c][input_y as usize]
                                        [input_x as usize];
                                let kernel_val = kernels[g * kernels_per_group + k][c][i][j];
                                sum += in_val * kernel_val;
                            }
                        }
                        output[g * kernels_per_group + k][y][x] += sum;
                    }
                }
            }
        }
    }
    output
}

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<i32>
        ::arange(0, 8 * 80 * 4 * 4)?
        .reshape([80, 8, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<i32>
        ::arange(0, 8 * 1260 * 1260)?
        .reshape([8, 1260, 1260])?
        .permute([1, 2, 0])?
        .contiguous()?;

    let now = std::time::Instant::now();
    for _ in 0..100 {
        let _ = conv2d_ex_i32(
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
