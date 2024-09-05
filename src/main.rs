use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(3);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 3 * 128 * 3 * 3)?
        .reshape([128, 3, 3, 3])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 100 * 3 * 128 * 128)?
        .reshape([100, 3, 128, 128])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let res = a.conv2d(
            &kernel,
            [2, 2],
            [
                (2, 2),
                (2, 2),
            ],
            [2, 2]
        )?;
        // println!("{:?}", res.permute([0, 3, 1, 2])?);
    }
    // println!("{:?}", res);
    println!("{:?}", now.elapsed() / 100);
    // let res2 = conv2d_naive(&a, &kernel, [1, 1])?;
    // println!("{:?}", res2);
    Ok(())
}
