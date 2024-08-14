use ops::cpu::convolutions::conv2d::{conv2d, conv2d_naive};
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(10);
    let kernel = _Tensor::<f32>::arange(0, 32)?.reshape([2, 1, 4, 4])?;
    let a = _Tensor::<f32>::arange(0, 500000)?.reshape([2, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = conv2d(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());

    println!("{:?}", res);

    let kernel = _Tensor::<f32>::arange(0, 32)?.reshape([1, 2, 4, 4])?;
    let a = _Tensor::<f32>::arange(0, 500000)?.reshape([2, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = conv2d_naive(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());

    println!("{:?}", res);
    Ok(())
}
