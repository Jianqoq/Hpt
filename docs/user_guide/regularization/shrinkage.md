# shrinkage
```rust
shrinkage(
    x: &Tensor<T>,
    bias: T,
    lambda: T
) -> Result<Tensor<T>, TensorError>
```
Applies the shrinkage function to the input tensor. The shrinkage function is a soft thresholding operator commonly used in signal processing and optimization algorithms, defined as:
sign(x - bias) * max(abs(x - bias) - lambda, 0)

## Parameters:
`x`: Input tensor containing indices.

`bias`: Bias value to subtract from each element before applying shrinkage.

`lambda`: Threshold parameter controlling the amount of shrinkage.

## Returns:
A new tensor with the same shape as input, where each element has been transformed by the shrinkage function.

## Examples:
```rust
use hpt::{Tensor, error::TensorError, ops::RegularizationOps};

fn main() -> Result<(), TensorError> {
    // Create a tensor
    let x = Tensor::<f32>::new(&[[-3.0, -1.0, 0.0, 2.0, 5.0]]);
    
    // Apply shrinkage with bias=0 and lambda=1.5
    let result = x.shrinkage(0.0, 1.5)?;
    println!("Shrinkage result:\n{}", result);
    // Output:
    // [[-1.5, 0.0, 0.0, 0.5, 3.5]]
    
    Ok(())
}
```