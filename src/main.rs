use ops::cpu::convolutions::conv2d::{conv2d, conv2d_naive};
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(10);
    let kernel = _Tensor::<f32>::arange(0, 480)?.reshape([3, 10, 4, 4])?.permute([2, 3, 0, 1])?.contiguous()?;
    let a = _Tensor::<f32>::arange(0, 750000)?.reshape([3, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = conv2d(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());

    // println!("{:?}", res);

    let kernel = _Tensor::<f32>::arange(0, 480)?.reshape([10, 3, 4, 4])?;
    let a = _Tensor::<f32>::arange(0, 750000)?.reshape([3, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = conv2d_naive(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());

    // println!("{:?}", res);
    Ok(())
}
