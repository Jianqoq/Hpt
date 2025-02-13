# exp10
```rust
exp10(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute 10 raised to the power of `x` for all elements

## Parameters:
`x`: Input values (exponents)

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);  // 10^2
    let b = a.exp10()?;
    println!("{}", b);  // prints: 100.0
    Ok(())
}
```