use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]); // 10^2
    let b = a.exp10_(&mut a.clone())?;
    println!("{}", b); // prints: 100.0
    Ok(())
}
