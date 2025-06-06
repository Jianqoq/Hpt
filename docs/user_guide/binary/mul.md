# mul
```rust
std::ops::Mul::mul(
    x: &Tensor<A> | Tensor<A> | scalar, 
    y: &Tensor<B> | Tensor<B> | scalar
) -> Result<Tensor<C>, TensorError>
```
Compute $\large x * y$ for all elements

## Parameters:
`x`: First input tensor

`y`: Second input tensor

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = &a * &a;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |