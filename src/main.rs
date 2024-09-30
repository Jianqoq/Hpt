// use std::ffi::CStr;
// use std::hint::black_box;
// use half::bf16;
use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

const IN: i64 = 8192;
const OUT: i64 = 256;
const KH: i64 = 3;
const KW: i64 = 3;
const H: i64 = 222;
const W: i64 = 240;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(10);
    println!("{:?}", cache_size::l2_cache_size().unwrap() / std::mem::size_of::<f32>());
    set_num_threads(16);
    let kernel = _Tensor::<f32>
        // ::arange(0, IN * 16 * 3 * 3)?
        // .reshape([IN, 16, 3, 3])?
        // .permute([1, 2, 3, 0])?
        ::arange(0, OUT * IN * KH * KW)?
        .reshape([OUT, IN, KH, KW])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * IN * H * W)?
        .reshape([1, IN, H, W])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<f32>::new(OUT, IN, [KH, KW], KernelParamAlgo::Greedy);
    // println!("config: {:?}", config);
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = a.iconv2d(
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            Some(&config)
        )?;
        println!("{}", res);
    }
    println!("{:?}", now.elapsed() / 5);

    // // println!("{:?}", res);
    Ok(())
}
