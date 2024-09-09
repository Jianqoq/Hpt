use std::ffi::CStr;

use half::bf16;
use ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(16);
    let kernel = _Tensor::<f32>
        ::arange(0, 8 * 80 * 4 * 4)?
        .reshape([80, 8, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * 8 * 1260 * 1260)?
        .reshape([1, 8, 1260, 1260])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<f32>::new(
        80,
        8,
        [4, 4],
        KernelParamAlgo::Greedy
    );
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let res = a.conv2d(
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
    // println!("{:?}", res);
    println!("{:?}", now.elapsed() / 100);
    // println!("{:?}", res.shape());
    // let res2 = conv2d_naive(
    //     &a,
    //     &kernel,
    //     [1, 1],
    // )?;
    // println!("{:?}", res2);
    // assert_eq!(res, res2);
    Ok(())
}
