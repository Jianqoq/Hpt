# log_softmax
```rust
log_softmax(
    x: &Tensor<T>,
    dim: i64
) -> Result<Tensor<C>, TensorError>
```
Applies the log-softmax function to the input tensor along the specified dimension. The log-softmax function is equivalent to applying the logarithm to the output of the softmax function, but is more numerically stable when computed directly.

## Parameters:
`x`: Input tensor.

`dim`: The dimension along which to apply the log-softmax.

## Returns:
A new tensor with the same shape as input with type `C`

## Examples:
```rust
use hpt::{Tensor, TensorError, NormalizationOps};

fn main() -> Result<(), TensorError> {
    // Create a 2x3 tensor
    let x = Tensor::<f32>::new(&[[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]);
    
    // Apply log-softmax along dimension 1 (columns)
    let result = x.log_softmax(1)?;
    println!("Log-softmax result:\n{}", result);
    
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |