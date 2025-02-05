# log2
```rust
log2(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \log_{2}(x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([8.0]);
    let b = a.log2()?;
    println!("{}", b);  // prints: 3.0
    Ok(())
}
```