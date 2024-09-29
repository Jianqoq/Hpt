// use std::ffi::CStr;
// use std::hint::black_box;
// use half::bf16;
use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

const IN: i64 = 1024;

fn main() -> anyhow::Result<()> {
    set_num_threads(16);
    let kernel = _Tensor::<f32>
        // ::arange(0, IN * 16 * 3 * 3)?
        // .reshape([IN, 16, 3, 3])?
        // .permute([1, 2, 3, 0])?
        ::arange(0, 1024 * IN * 3 * 3)?
        .reshape([1024, IN, 3, 3])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * IN * 224 * 224)?
        .reshape([1, IN, 224, 224])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<f32>::new(512, IN, [3, 3], KernelParamAlgo::Greedy);
    // println!("config: {:?}", config);
    let now = std::time::Instant::now();
    for _ in 0..5 {
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
    }
    println!("{:?}", now.elapsed() / 5);

    // // println!("{:?}", res);
    Ok(())
}
