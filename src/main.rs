use ops::cpu::convolutions::conv2d::conv2d_pad_dilation_group;
use ops::cpu::convolutions::conv2d_unroll::{
    conv2d_block_simd_parallel_unroll_f32,
    conv2d_group,
};
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use wide::i32x8;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<i32>
        ::arange(0, 10240)?
        .reshape([8, 80, 4, 4])?
        .permute([2, 3, 0, 1])?
        .contiguous()?;
    let a = _Tensor::<i32>
        ::arange(0, 8 * 256 * 1260)?
        .reshape([8, 256, 1260])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let now = std::time::Instant::now();
    let res1: _Tensor<i32> = conv2d_pad_dilation_group(
        &a,
        &kernel,
        [2, 2],
        [
            (2, 2),
            (2, 2),
        ],
        [2, 2],
        2
    )?.permute([2, 0, 1])?;

    let kernel = _Tensor::<i32>
        ::arange(0, 10240)?
        .reshape([8, 80, 4, 4])?
        .permute([2, 3, 0, 1])?
        .contiguous()?;
    let a = _Tensor::<i32>
        ::arange(0, 8 * 256 * 1260)?
        .reshape([8, 256, 1260])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res2 = conv2d_group::<i32, i32x8, false, 14, true>(
            &a,
            &kernel,
            [2, 2],
            [
                (2, 2),
                (2, 2),
            ],
            [2, 2],
            2
        )?.permute([2, 0, 1])?;
        // assert_eq!(res1, res2);
    }
    println!("{:?}", now.elapsed() / 100);

    Ok(())
}
