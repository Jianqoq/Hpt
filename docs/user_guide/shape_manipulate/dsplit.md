# dsplit
```rust
dsplit(
    x: &Tensor<T>,
    indices: &[i64]
) -> Result<Vec<Tensor<T>>, TensorError>
```
Splits a tensor into multiple sub-tensors along axis 2 (depth). The tensor must be at least 3-dimensional.

## Parameters:
`x`: Input tensor with ndim >= 3

`indices`: The indices where the splits should occur along axis 2

## Returns:
A vector of sub-tensors created by splitting the input tensor along the depth axis.

## Examples:
```rust
use tensor_dyn::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 3D tensor with shape [2, 2, 4]
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0]).reshape(&[2, 2, 4])?;
    
    // Split along depth (axis 2) at index [2]
    let splits = a.dsplit(&[2])?;
    // splits[0]: shape [2, 2, 2]
    // [[[1, 2],
    //   [5, 6]],
    //  [[9, 10],
    //   [13, 14]]]
    //
    // splits[1]: shape [2, 2, 2]
    // [[[3, 4],
    //   [7, 8]],
    //  [[11, 12],
    //   [15, 16]]]
    for (i, split) in splits.iter().enumerate() {
        println!("Split {}: {}", i, split);
    }

    // Will raise an error for 2D tensor
    let b = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    assert!(b.dsplit(&[1]).is_err());

    Ok(())
}
```