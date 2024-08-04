
use tensor_dyn::tensor::Tensor;
use tensor_traits::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0.0, 9.0)?.reshape(&[3, 3])?;
    println!("{:?}", a);
    let (a, b) = a.topk(2, 0, false, false)?;
    println!("{:?}", a);
    println!("{:?}", b);
    Ok(())
}
