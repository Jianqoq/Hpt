use hpt::{error::TensorError, ops::WindowOps, Tensor};
fn main() -> Result<(), TensorError> {
    // Create a periodic Hann window of length 5
    let a = Tensor::<f32>::hann_window(5, true)?;
    println!("{}", a);
    // [0.0000, 0.3455, 0.9045, 0.9045, 0.0000]

    // Create a symmetric Hann window of length 5
    let b = Tensor::<f32>::hann_window(5, false)?;
    println!("{}", b);
    // [0.0000, 0.5000, 1.0000, 0.5000, 0.0000]

    Ok(())
}
