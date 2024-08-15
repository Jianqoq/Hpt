use ops::cpu::convolutions::conv2d::{
    conv2d,
    conv2d_block,
    conv2d_block_parallel,
    conv2d_block_simd,
    conv2d_block_simd_parallel,
    conv2d_naive,
};
use ops::cpu::convolutions::conv2d_unroll::conv2d_block_simd_parallel_unroll;
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    // let kernel = _Tensor::<f32>
    //     ::arange(0, 10240)?
    //     .reshape([8, 80, 4, 4])?
    //     .permute([2, 3, 0, 1])?
    //     .contiguous()?;
    // let a = _Tensor::<f32>
    //     ::arange(0, 8 * 256 * 128)?
    //     .reshape([8, 256, 128])?
    //     .permute([1, 2, 0])?
    //     .contiguous()?;
    // let now = std::time::Instant::now();
    // let res1 = conv2d_block(&a, &kernel, [1, 1])?.permute([2, 0, 1])?;
    // println!("{:?}", now.elapsed());
    let kernel = _Tensor::<f32>
        ::arange(0, 8960)?
        .reshape([8, 70, 4, 4])?
        .permute([2, 3, 0, 1])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 8 * 256 * 128)?
        .reshape([8, 256, 128])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let now = std::time::Instant::now();
    let res = conv2d_block_simd_parallel(&a, &kernel, [1, 1])?.permute([2, 0, 1])?;

    println!("{:?}", now.elapsed());
    let now = std::time::Instant::now();
    let res2 = conv2d_block_simd_parallel_unroll(&a, &kernel, [1, 1])?.permute([2, 0, 1])?;
    println!("{:?}", now.elapsed());

    println!("{:?}", res);
    println!("{:?}", res2);
    Ok(())
}
