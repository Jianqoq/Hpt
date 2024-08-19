use ops::cpu::convolutions::conv2d_revised::conv2d_ex_f32_revised;
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(1);
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

    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = conv2d_ex_f32_revised::<8, 7, 14>(
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
    println!("{:?}", now.elapsed() / 1);
    Ok(())
}
