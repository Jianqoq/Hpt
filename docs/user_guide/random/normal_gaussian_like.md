# normal_gaussian_like
```rust
normal_gaussian_like(
    x: &Tensor<T>,
    mean: T,
    std: T
) -> Result<Tensor<T>, TensorError>
```
Same as `normal_gaussian` but the shape will be based on `x`. Creates a Tensor with values drawn from a normal distribution with specified mean and standard deviation.

## Parameters:
`x`: Input Tensor to derive the shape from

`mean`: Mean (μ) of the distribution, determining the center of the bell curve.

`std`: Standard deviation (σ) of the distribution, determining the spread. Must be positive.

## Returns:
Tensor with type `T` containing random values from the normal distribution.

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with normal distribution
    let n = x.normal_gaussian_like(0.0, 1.0)?;
    println!("{}", n);
    Ok(())
}
```