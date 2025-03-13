use hpt::{Tensor, backend::Cuda};
use hpt::ops::FloatUnaryOps;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]).to_cuda::<0/*Cuda device id*/>()?;
    let y = Tensor::<i64>::new(&[4i64, 5, 6]).to_cuda::<0/*Cuda device id*/>()?;

    let result = x + &y; // with `normal_promote` feature enabled, i64 + f64 will output f64
    println!("{}", result); // [5. 7. 9.]

    // All the available methods are listed in https://jianqoq.github.io/Hpt/user_guide/user_guide.html
    let result: Tensor<f64, Cuda, 0> = y.sin()?;
    println!("{}", result); // [-0.7568 -0.9589 -0.2794]
    Ok(())
}