use tensor_dyn::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [2, 3, 4]
    let a = Tensor::<f32>::zeros(&[2, 3, 4])?;

    // Permute dimensions to [4, 2, 3]
    let b = a.permute(&[2, 0, 1])?;
    println!("{}", b.shape());

    // Permute dimensions to [3, 4, 2]
    let c = a.permute(&[1, 2, 0])?;
    println!("{}", c.shape());

    // This will return an error as [1, 1, 0] is not a valid permutation
    let d = a.permute(&[1, 1, 0]);
    assert!(d.is_err());

    Ok(())
}