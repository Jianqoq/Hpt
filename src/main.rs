use tensor_dyn::TensorCreator;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::slice::SliceOps;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(10);
    let kernel = _Tensor::<f32>
        ::new([
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
        ])
        .reshape([2, 9])?
        .t()?; // shape [9, 2]
    println!("{:?}", kernel);
    let a = _Tensor::<f32>::arange(0, 10000000)?.reshape([20, 2, 500, 500])?; // shape [2, 2, 5, 5]
    let now = std::time::Instant::now();
    // let res = a.img2col(
    //     &[1, 3, 3],
    //     Some(&[1, 1]),
    //     Some(
    //         &[
    //             (0, 0),
    //             (0, 0),
    //         ]
    //     ),
    //     Some(&[1, 1]),
    //     Some(2)
    // )?; // shape [2, 2 * 3 * 3, 1 * 3 * 3]
    // let start = res.shape()[1] / 2;
    // let group1 = slice!(res[:, :start, :])?;
    // let group2 = slice!(res[:, start:, :])?;
    // let kernel_sliced1 = slice!(kernel[:, 0])?;
    // let kernel_sliced2 = slice!(kernel[:, 1])?;
    // let res1 = group1.matmul(&kernel_sliced1)?; // shape [2, 3 * 3, 1]
    // let res2 = group2.matmul(&kernel_sliced2)?; // shape [2, 3 * 3, 1]
    // let res = _Tensor::<f32>::stack(vec![&res1, &res2], 2, true)?; // shape [2, 2 * 3 * 3, 2]

    println!("{:?}", now.elapsed());
    Ok(())
}
