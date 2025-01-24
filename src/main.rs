use tensor_dyn::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [1, 3, 1, 4]
    let a = Tensor::<f32>::zeros(&[1, 3, 1, 4])?;

    // Remove single-dimensional entry at axis 0
    let b = a.squeeze(0)?; // shape becomes [3, 1, 4]
    println!("{}", b.shape());
    // Remove single-dimensional entry at axis 2
    let c = a.squeeze(2)?; // shape becomes [1, 3, 4]
    println!("{}", c.shape());

    Ok(())
}
