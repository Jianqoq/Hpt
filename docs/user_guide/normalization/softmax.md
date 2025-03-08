# softmax
```rust
softmax(
    x: &Tensor<T>,
    dim: i64
) -> Result<Tensor<C>, TensorError>
```
Applies the softmax function to the input tensor along the specified dimension. The softmax function normalizes the input to a probability distribution, such that each element is in the range [0, 1] and all elements sum to 1.

## Parameters:
`x`: Input tensor.

`dim`: The dimension along which to apply the softmax.

## Returns:
A new tensor with the same shape as input with type `C`

## Examples:
```rust
use hpt::{error::TensorError, ops::NormalizationOps, Tensor};

fn main() -> Result<(), TensorError> {
    // Create a 2x3 tensor
    let x = Tensor::<f32>::new(&[[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]);

    // Apply softmax along dimension 1 (columns)
    let result = x.softmax(1)?;
    println!("Softmax result:\n{}", result);

    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |