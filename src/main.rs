use tensor_dyn::tensor::Tensor;
use tensor_traits::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0.0, 30.0)?.reshape(&[2, 3, 5])?;
    let b = Tensor::<f32>::arange(0.0, 30.0)?.reshape(&[2, 3, 5])?;
    a.sin_(&b)?;
    Ok(())
}
