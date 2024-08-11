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
    ]);
    let a = _Tensor::<f32>::arange(0, 50000000)?.reshape([200, 1, 500, 500])?;
    let mut i = 0;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let _ = a.conv(&kernel, None, None, None, None, None, None, None)?;
        i += 1;
    }
    println!("{:?}", now.elapsed());
    println!("{}", i);
    Ok(())
}
