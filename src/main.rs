use tensor_dyn::{FloatUaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.sin_(&a)?;
    println!("{}", b);
    Ok(())
}
