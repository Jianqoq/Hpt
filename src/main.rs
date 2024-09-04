use ops::cpu::convolutions::conv2d::{ conv2d_naive, conv2d_naive_f32 };
use ops::cpu::convolutions::conv2d_unroll::{ conv2d_ex, conv2d_ex_naive };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use tensor_dyn::f32x8::f32x8;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 3 * 80 * 4 * 4)?
        .reshape([80, 3, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * 3 * 1024 * 1024)?
        .reshape([1, 3, 1024, 1024])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = conv2d_ex::<f32, 7, 8, f32x8>(
            &a,
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1]
        )?;
        println!("{:?}", res);
    }
    // println!("{:?}", res);
    println!("{:?}", now.elapsed() / 100);
    let res2 = conv2d_naive(&a, &kernel, [1, 1])?;
    println!("{:?}", res2);
    Ok(())
}
