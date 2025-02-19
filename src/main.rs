use hpt::{TensorCmp, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    let b = a.tensor_eq(&a)?;
    println!("{}", b); // [true true true]
    Ok(())
}