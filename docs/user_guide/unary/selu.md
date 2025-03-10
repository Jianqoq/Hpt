# selu
```rust
selu(
    x: &Tensor<T>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large \lambda * (\alpha * (e^x - 1))$ for $x < 0$, $\large \lambda * x$ for $x \geq 0$ for all elements

where `alpha` is `1.6732632423543772848170429916717`, `gamma` is `1.0507009873554804934193349852946`

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.selu()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |