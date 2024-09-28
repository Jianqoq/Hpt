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
        ::arange(0, 16 * 1024 * 3 * 3)?
        .reshape([16, 1024, 3, 3])?
        .permute([1, 2, 3, 0])?
        // ::arange(0, 16 * 16 * 3 * 3)?
        // .reshape([16, 16, 3, 3])?
        // .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 16 * 16 * 256 * 252)?
        .reshape([16, 16, 256, 252])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<f32>::new(16, 16, [3, 3], KernelParamAlgo::Greedy);
    // println!("config: {:?}", config);
    let now = std::time::Instant::now();
    for _ in 0..10 {
        let _: _Tensor<f32> = a
            .iconv2d(
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