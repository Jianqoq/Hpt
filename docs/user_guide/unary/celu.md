# celu
```rust
celu(x: &Tensor<T>, alpha: C) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{max}(0, x) + \text{min}(0, \alpha \cdot (e^{x/\alpha} - 1))$ for all elements

## Parameters:
`x`: Input values

`alpha`: Parameter controlling the saturation of negative values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-1.0]);
    let b = a.celu(1.0)?;
    println!("{}", b);  // prints: -0.6321206
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |