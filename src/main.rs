// use std::ffi::CStr;
// use std::hint::black_box;
// use half::bf16;
use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

const IN: i64 = 4096;

fn main() -> anyhow::Result<()> {
    set_num_threads(16);
    let kernel = _Tensor::<f32>
        ::arange(0, IN * 16 * 3 * 3)?
        .reshape([IN, 16, 3, 3])?
        .permute([1, 2, 3, 0])?
        // ::arange(0, 16 * 16 * 3 * 3)?
        // .reshape([16, 16, 3, 3])?
        // .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * IN * 256 * 254)?
        .reshape([1, IN, 256, 254])?
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
    println!("{:?}", now.elapsed() / 10);
    
    // // println!("{:?}", res);
    Ok(())
}