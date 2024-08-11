use tensor_dyn::{ set_global_display_precision, TensorCreator };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    let kernel = _Tensor::<f32>::new(
        [[[[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]]]]
    );
    let image = _Tensor::<f32>::arange(0, 50000000)?.reshape(&[200, 1, 500, 500])?;
    println!("{}", image.shape());
    println!("{}", kernel.shape());
    let now = std::time::Instant::now();
    let out = image.conv(&kernel, None, None, None, None, None, None, None)?;
    println!("{:?}", now.elapsed());
    println!("{}", out);
    Ok(())
}
