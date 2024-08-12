use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(10);
    let kernel = _Tensor::<f32>::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
    ]);
    println!("{:?}", kernel);
    let a = _Tensor::<f32>::arange(0, 100)?.reshape([2, 2, 5, 5])?;
    println!("{:?}", a);
    let now = std::time::Instant::now();
    let res = a.img2col(
        kernel.shape(),
        Some(&[1, 1]),
        Some(
            &[
                (0, 0),
                (0, 0),
            ]
        ),
        Some(&[1, 1]),
        Some(1)
    )?;
    println!("{:?}", now.elapsed());
    println!("{:?}", res);
    Ok(())
}
