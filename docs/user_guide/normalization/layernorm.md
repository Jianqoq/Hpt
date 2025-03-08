# softmax
```rust
layernorm<S: Into<Shape>>(
    &self,
    normalized_shape: S,
    gamma: Option<&Tensor<T>>,
    beta: Option<&Tensor<T>>,
    eps: T
) -> Result<Tensor<T>, TensorError>
```
Applies Layer Normalization over a specified axes.

## Parameters:
`normalized_shape`: shape that must match the dimension size from input tensor shape (from right to left)

`gamma`: Optional scale tensor of shape `[normalized_shape]`

`beta`: Optional bias tensor of shape `[normalized_shape]`

`eps`: A value added to the denominator for numerical stability.

## Returns:
A new tensor with the same shape as input with type `C`

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{NormalizationOps, Random, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // Create a 2x3x4 tensor
    let x = Tensor::<f32>::randn(&[2, 3, 4])?;

    // Create optional gamma and beta parameters
    let gamma = Tensor::<f32>::ones(&[4])?;
    let beta = Tensor::<f32>::zeros(&[4])?;

    // Apply layer normalization over the last dimension
    let result = x.layernorm(&[4], Some(&gamma), Some(&beta), 1e-5)?;
    println!("LayerNorm result shape: {:?}", result.shape());

    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |