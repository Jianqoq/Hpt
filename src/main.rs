
use ops::cpu::convolutions::conv2d_unroll::conv2d_ex;
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use tensor_dyn::f32x8::f32x8;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 8 * 80 * 4 * 4)?
        .reshape([80, 8, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * 8 * 1027 * 1027)?
        .reshape([1, 8, 1027, 1027])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = conv2d_ex::<f32, 14, 8, f32x8>(
            &a,
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1]
        )?.permute([0, 3, 1, 2])?;
        // println!("{:?}", res);
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
