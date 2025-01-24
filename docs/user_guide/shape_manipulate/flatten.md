# flatten
```rust
flatten(
    x: &Tensor<T>,
    start_dim: Option<usize>,
    end_dim: Option<usize>
) -> Result<Tensor<T>, TensorError>
```
Flattens a contiguous range of dimensions in a tensor into a single dimension.

## Parameters:
`x`: Input tensor

`start_dim`: Starting dimension to flatten (inclusive). Defaults to 0 if None

`end_dim`: Ending dimension to flatten (inclusive). Defaults to last dimension if None

## Returns:
A new tensor with the specified dimensions flattened into one.

## Examples:
```rust
use tensor_dyn::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 3D tensor with shape [2, 3, 2]
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                                .reshape(&[2, 3, 2])?;
    
    // Flatten all dimensions (default behavior)
    let b = a.flatten(None, None)?;
    // Shape: [12]
    println!("{}", b);

    // Flatten dimensions 1 and 2
    let c = a.flatten(Some(1), Some(2))?;
    // Shape: [2, 6]
    println!("{}", c);

    // Flatten first two dimensions
    let d = a.flatten(Some(0), Some(1))?;
    // Shape: [6, 2]
    println!("{}", d);

    // Will raise an error for invalid dimensions
    assert!(a.flatten(Some(0), Some(3)).is_err());

    Ok(())
}
```