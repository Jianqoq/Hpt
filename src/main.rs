
use ops::cpu::convolutions::conv2d::conv2d_pad_dilation;
use ops::cpu::convolutions::conv2d_unroll::conv2d_ex_f32;
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 8 * 80 * 4 * 4)?
        .reshape([80, 8, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 8 * 1260 * 1260)?
        .reshape([8, 1260, 1260])?
        .permute([1, 2, 0])?
        .contiguous()?;

    // let c = conv2d_pad_dilation(
    //     &a,
    //     &kernel,
    //     [1, 1],
    //     [
    //         (2, 2),
    //         (2, 2),
    //     ],
    //     [2, 2]
    // )?.permute([2, 0, 1])?;

    let now = std::time::Instant::now();
    for _ in 0..100 {
        let res = conv2d_ex_f32(
            &a,
            &kernel,
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [2, 2]
        )?.permute([2, 0, 1])?;
        
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
