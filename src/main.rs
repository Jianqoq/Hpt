use ops::cpu::convolutions::conv2d::{ conv2d, conv2d_naive, conv2d_block };
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    let kernel = _Tensor::<i32>
        ::arange(0, 1024)?
        .reshape([8, 8, 4, 4])?
        .permute([2, 3, 0, 1])?
        .contiguous()?;
    let a = _Tensor::<i32>
        ::arange(0, 8 * 256 * 128)?
        .reshape([8, 256, 128])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let now = std::time::Instant::now();
    let res1 = conv2d_block(&a, &kernel, [1, 1])?.permute([2, 0, 1])?;
    println!("{:?}", now.elapsed());

    let kernel = _Tensor::<i32>
        ::arange(0, 1024)?
        .reshape([8, 8, 4, 4])?
        .permute([2, 3, 0, 1])?
        .contiguous()?;
    let a = _Tensor::<i32>
        ::arange(0, 8 * 256 * 128)?
        .reshape([8, 256, 128])?
        .permute([1, 2, 0])?
        .contiguous()?;
    let now = std::time::Instant::now();
    let _ = conv2d(&a, &kernel, [1, 1])?.permute([2, 0, 1])?;
    println!("{:?}", now.elapsed());

    let kernel = _Tensor::<i32>
        ::arange(0, 1024)?
        .reshape([8, 8, 4, 4])?
        .permute([1, 0, 2, 3])?
        .contiguous()?;
    let a = _Tensor::<i32>::arange(0, 8 * 256 * 128)?.reshape([8, 256, 128])?;
    let now = std::time::Instant::now();
    let res2 = conv2d_naive(&a, &kernel, [1, 1])?;
    println!("{:?}", now.elapsed());

    assert!(res1.allclose(&res2));
    Ok(())
}
