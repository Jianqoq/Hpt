# split
```rust
split(
    x: &Tensor<T>,
    indices_or_sections: &[i64],
    axis: i64
) -> Result<Vec<Tensor<T>>, TensorError>
```
Splits a tensor into multiple sub-tensors along a specified axis at given indices.

## Parameters:
`x`: Input tensor

`indices_or_sections`: The indices where the splits should occur

`axis`: The axis along which to split the tensor. Negative values count from the end

## Returns:
A vector of sub-tensors created by splitting the input tensor.

## Examples:
```rust
use hpt::{ops::ShapeManipulate, Tensor, error::TensorError};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [6]
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Split at indices [2, 4]
    let splits = a.split(&[2, 4], 0)?;
    // splits[0]: [1, 2]
    // splits[1]: [3, 4]
    // splits[2]: [5, 6]
    for (i, split) in splits.iter().enumerate() {
        println!("Split {}: {}", i, split);
    }

    // Create a 2D tensor with shape [4, 3]
    let b = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 
                                4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0,
                                10.0, 11.0, 12.0]).reshape(&[4, 3])?;
    
    // Split along axis 0 at index [2]
    let splits = b.split(&[2], 0)?;
    // splits[0]:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // splits[1]:
    // [[7, 8, 9],
    //  [10, 11, 12]]
    for (i, split) in splits.iter().enumerate() {
        println!("Split {}: {}", i, split);
    }

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |