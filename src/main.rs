use ops::cpu::convolutions::conv2d::{conv2d, conv2d_naive};
use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::slice::SliceOps;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(10);
    let kernel = _Tensor::<f32>::new([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
        ],
    ]);
    let a = _Tensor::<f32>::arange(0, 500000)?.reshape([1, 2, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = a.conv(
        &kernel,
        None,
        None,
        None,
        Some(
            &[
                (0, 0),
                (0, 0),
            ]
        ),
        Some(&[1, 1]),
        Some(1)
    )?; // shape [2, 2 * 3 * 3, 1 * 3 * 3]
    println!("{:?}", now.elapsed());
    println!("{:?}", res);

    let kernel = _Tensor::<f32>::new([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
        ],
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
        ],
    ]);
    let a = _Tensor::<f32>::arange(0, 500000)?.reshape([2, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = conv2d(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());

    println!("{:?}", res);

    let kernel = _Tensor::<f32>::new([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
        ],
    ]);
    let a = _Tensor::<f32>::arange(0, 500000)?.reshape([2, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    let res = conv2d_naive(&a, &kernel, [1, 1])?;

    println!("{:?}", now.elapsed());

    println!("{:?}", res);
    Ok(())
}
