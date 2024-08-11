use tensor_dyn::{ set_global_display_precision, TensorCreator };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
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
    let a = _Tensor::<f32>::arange(0, 100)?.reshape([2, 2, 5, 5])?;
    let now = std::time::Instant::now();
    let res = a.conv(&kernel, None, None, Some(&[2, 2]), Some(&[(1, 1), (1, 1)]), Some(&[2, 2]), None, Some(2))?;
    println!("{:?}", now.elapsed());
    println!("{}", res);
    Ok(())
}
