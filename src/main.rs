use ops::cpu::convolutions::conv2d::{
    conv2d_pad_dilation,
    conv2d_pad_dilation_ex,
    conv2d_pad_dilation_group,
};
use ops::cpu::convolutions::conv2d_unroll::{ conv2d_block_simd_parallel_unroll_f32, conv2d_ex_f32, conv2d_group, conv2d_no_group };
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use wide::{f32x8, i32x8};

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 8 * 80 * 4 * 4)?
        .reshape([80, 8, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 8 * 256 * 1260)?
        .reshape([8, 256, 1260])?
        .permute([1, 2, 0])?
        .contiguous()?;
    // let res1 = conv2d_pad_dilation(
    //     &a,
    //     &kernel,
    //     [1, 1],
    //     [
    //         (1, 1),
    //         (1, 1),
    //     ],
    //     [1, 1]
    // )?.permute([2, 0, 1])?;

    let now = std::time::Instant::now();
    for _ in 0..1 {
        let _ = conv2d_ex_f32::<f32, f32x8, true, 14>(
            &a,
            &kernel,
            [1, 1],
            [
                (1, 1),
                (1, 1),
            ],
            [1, 1]
        )?.permute([2, 0, 1])?;
    }
    println!("{:?}", now.elapsed());

    Ok(())
}
