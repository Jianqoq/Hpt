use ops::cpu::convolutions::conv2d::{ conv2d, conv2d_naive };
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    let kernel = _Tensor::<f32>
        ::arange(0, 480)?
        .reshape([3, 10, 4, 4])?
        .permute([2, 3, 0, 1])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 3 * 256 * 128)?
        .reshape([3, 256, 128])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let now = std::time::Instant::now();
    let res = conv2d(&a, &kernel, [1, 1])?.permute([2, 0, 1])?;

    println!("{:?}", now.elapsed());

    let kernel = _Tensor::<f32>
        ::arange(0, 480)?
        .reshape([3, 10, 4, 4])?
        .permute([1, 0, 2, 3])?
        .contiguous()?;
    let a = _Tensor::<f32>::arange(0, 3 * 256 * 128)?.reshape([3, 256, 128])?;
    let now = std::time::Instant::now();
    let res = conv2d_naive(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());
    Ok(())
}
