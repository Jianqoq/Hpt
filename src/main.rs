
use tensor_dyn::tensor::Tensor;
use tensor_traits::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<i8>::arange(0.0, 9.0)?.reshape(&[3, 3])?;
    let res = a.logsumexp(1, false)?;
    println!("{:?}", res);
    Ok(())
}
