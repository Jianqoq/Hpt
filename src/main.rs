
use tensor_dyn::{Cuda, Tensor, TensorCreator};
use tensor_dyn::NormalReduce;


fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32, Cuda, 0>::ones([10, 10])?;

    let now = std::time::Instant::now();
    let _ = a.reducel1(0, false)?;
    println!("{:?}", now.elapsed());
    Ok(())
}
