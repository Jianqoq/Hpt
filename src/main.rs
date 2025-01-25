use tensor_dyn::{Tensor, TensorError, WindowOps};
fn main() -> Result<(), TensorError> {
    // Create a periodic Blackman window of length 5
    let a = Tensor::<f32>::blackman_window(5, true)?;
    println!("{}", a);
    // [-0.0000, 0.2008, 0.8492, 0.8492, 0.2008]

    // Create a symmetric Blackman window of length 5
    let b = Tensor::<f32>::blackman_window(5, false)?;
    println!("{}", b);
    // [-0.0000, 0.3400, 1.0000, 0.3400]

    Ok(())
}
