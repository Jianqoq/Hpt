# concat
```rust
concat(
    tensors: Vec<&Tensor<T>>,
    axis: usize,
    keepdims: bool
) -> Result<Tensor<T>, TensorError>
```
Concatenates a sequence of tensors along the specified axis.

## Parameters:
`tensors`: Vector of tensors to concatenate

`axis`: The axis along which to concatenate the tensors

`keepdims`: If true, inserts a new dimension at the concatenation axis, splitting the concatenated dimension into [num_tensors, concatenated_size]

## Returns:
A new tensor containing all the input tensors concatenated along the specified axis.

## Examples:
```rust
use tensor_dyn::{ShapeManipulate, Tensor, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create two 2D tensors
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    // [[1, 2],
    //  [3, 4]]
    let b = Tensor::<f32>::new(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2])?;
    // [[5, 6],
    //  [7, 8]]

    // Concatenate along axis 0 (vertically)
    let c = Tensor::concat(vec![&a, &b], 0, false)?;
    // [[1, 2],
    //  [3, 4],
    //  [5, 6],
    //  [7, 8]]
    println!("{}", c);

    // Concatenate along axis 1 (horizontally)
    let d = Tensor::concat(vec![&a, &b], 1, false)?;
    // [[1, 2, 5, 6],
    //  [3, 4, 7, 8]]
    println!("{}", d);

    // Concatenate with keepdims=true
    let e = Tensor::concat(vec![&a, &b], 0, true)?;
    // Shape: [2, 2, 2]
    println!("{}", e.shape());

    // Will raise an error if shapes don't match along non-concatenating axes
    let f = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3, 1])?;
    assert!(Tensor::concat(vec![&a, &f], 1, false).is_err());

    Ok(())
}
```