use half::bf16;
use ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(3);
    set_num_threads(10);
    tch::set_num_threads(10);
    let kernel = _Tensor::<half::f16>
        ::arange(0, 3 * 128 * 3 * 3)?
        .reshape([128, 3, 3, 3])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<half::f16>
        ::arange(0, 1 * 3 * 128 * 128)?
        .reshape([1, 3, 128, 128])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<half::f16>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
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
        )?.permute([0, 3, 1, 2])?;
    }
    println!("{:?}", now.elapsed() / 100);
    let tch_kernel = tch::Tensor
        ::arange(3 * 128 * 3 * 3, (tch::Kind::Half, tch::Device::Cpu))
        .reshape(&[128, 3, 3, 3]);
    let tch_a = tch::Tensor
        ::arange(1 * 3 * 128 * 128, (tch::Kind::Half, tch::Device::Cpu))
        .reshape(&[1, 3, 128, 128]);
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
