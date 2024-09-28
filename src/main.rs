// use std::ffi::CStr;
// use std::hint::black_box;

use std::hint::black_box;

// use half::bf16;
use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

const KERNEL_HEIGHT: usize = 5;
const KERNEL_WIDTH: usize = 5;
const IN_CHANNEL: usize = 4096;
const OUT_CHANNEL: usize = 16;
const IMG_HEIGHT: usize = 256;
const IMG_WIDTH: usize = 256;
const BATCH: usize = 1;

fn main() -> anyhow::Result<()> {
    set_num_threads(8);
    let kernel = _Tensor::<f32>
        ::randn(&[OUT_CHANNEL * IN_CHANNEL * KERNEL_HEIGHT * KERNEL_WIDTH])?
        .reshape([OUT_CHANNEL as i64, IN_CHANNEL as i64, KERNEL_HEIGHT as i64, KERNEL_WIDTH as i64])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::randn( &[BATCH * IN_CHANNEL * IMG_HEIGHT * IMG_WIDTH])?
        .reshape([BATCH as i64, IN_CHANNEL as i64, IMG_HEIGHT as i64, IMG_WIDTH as i64])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let mut config = Conv2dConfig::<f32>::new(OUT_CHANNEL as i64, IN_CHANNEL as i64, [KERNEL_HEIGHT as i64, KERNEL_WIDTH as i64], KernelParamAlgo::Greedy);
    // println!("config: {:?}", config);
    {
        let now = std::time::Instant::now();
    config.set_ci_block_size(4096 as i64);
    // config.set_co_block_size(16);
    for _ in 0..5 {
        println!("running");
        let res = a
            .conv2d(
                &kernel,
                [1, 1],
                [
                    (0, 0),
                    (0, 0),
                ],
                [1, 1],
                Some(&config)
            )?;
        // println!("{:?}", res.shape());
    }
    println!("{:?}", now.elapsed() / 5);
    };
    
    // // println!("{:?}", res);
    Ok(())
}