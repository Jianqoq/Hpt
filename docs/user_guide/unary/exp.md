# exp
```rust
exp(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute exponential of `x` for all elements
## Parameters:
`x`: Input values
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp()?;
    println!("{}", b);
    Ok(())
}
```