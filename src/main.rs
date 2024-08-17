use ops::cpu::convolutions::conv2d::{
    conv2d_pad_dilation,
    conv2d_pad_dilation_ex,
    conv2d_pad_dilation_group,
};
use ops::cpu::convolutions::conv2d_unroll::{ conv2d_block_simd_parallel_unroll_f32, conv2d_group };
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use wide::i32x8;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<i32>
        ::arange(0, 3 * 2 * 4 * 4)?
        .reshape([2, 3, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<i32>
        ::arange(0, 3 * 5 * 5)?
        .reshape([3, 5, 5])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let res1: _Tensor<i32> = conv2d_pad_dilation(
        &a,
        &kernel,
        [1, 1],
        [
            (2, 2),
            (2, 2),
        ],
        [1, 1]
    )?.permute([2, 0, 1])?;

    let res2 = conv2d_pad_dilation_ex(
        &a,
        &kernel,
        [1, 1],
        [
            (2, 2),
            (2, 2),
        ],
        [1, 1]
    )?.permute([2, 0, 1])?;
    assert_eq!(res1, res2);
    println!("{res1}");

    Ok(())
}
