use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

const IN: i64 = 128;
const OUT: i64 = 4096;
const KH: i64 = 3;
const KW: i64 = 3;
const H: i64 = 256;
const W: i64 = 256;

fn main() -> anyhow::Result<()> {
    // set_global_display_lr_elements(10);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, OUT * IN * KH * KW)?
        .reshape([OUT, IN, KH, KW])?
        // .permute([0, 2, 3, 1])?
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
    for _ in 0..5 {
        
        let _ = a.iconv2d(
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            Some(&config)
        )?;
        //  println!("{:?}", res);
    }
    println!("{:?}", now.elapsed() / 5);

    // // println!("{:?}", res);
    Ok(())
}
