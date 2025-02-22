use hpt::{Cuda, IndexReduce, Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let shape = [1, 256, 1];
    let a = Tensor::<f64, Cuda>::randn(&shape)?;
    let a = a.argmax(0, true)?;
    println!("{}", a);
    Ok(())
}
