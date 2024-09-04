use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(3);
    set_num_threads(10);
    let kernel = _Tensor::<f32>
        ::arange(0, 3 * 80 * 4 * 4)?
        .reshape([80, 3, 4, 4])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 10 * 3 * 1024 * 1024)?
        .reshape([10, 3, 1024, 1024])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let res = a.conv2d(
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1]
        )?;
        // println!("{:?}", res.permute([0, 3, 1, 2])?);
    }
    // println!("{:?}", res);
    println!("{:?}", now.elapsed() / 100);
    // let res2 = conv2d_naive(&a, &kernel, [1, 1])?;
    // println!("{:?}", res2);
    Ok(())
}
