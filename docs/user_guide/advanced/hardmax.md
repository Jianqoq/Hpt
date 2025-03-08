# hardmax
```rust
hardmax(
    x: &Tensor<T>,
    axis: i64
) -> Result<Tensor<T>, TensorError>
```
Applies the hardmax function to the input tensor along the specified dimension. The hardmax function sets the largest element along the specified dimension to 1 and all other elements to 0.

## Parameters:
`x`: Input tensor containing indices.

`axis`: The dimension along which to apply the hardmax.

## Returns:
A new tensor with the same shape as input, where the largest value along the specified dimension is set to 1 and all other values are set to 0.

## Examples:
```rust
use hpt::{Tensor, error::TensorError, ops::HardMax};

fn main() -> Result<(), TensorError> {
    // Create a 2x3 tensor
    let x = Tensor::<f32>::new(&[[-1.0, 0.0, 3.0], [2.0, 1.0, 4.0]]);
    
    // Apply hardmax along dimension 1 (columns)
    let result = x.hardmax(1)?;
    println!("Hardmax result:\n{}", result);
    // Output:
    // [[0, 0, 1],
    //  [0, 0, 1]]
    
    Ok(())
}
```