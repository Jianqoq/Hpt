# exp2
```rust
exp2(x: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large 2^x$ for all elements
## Parameters:
`x`: Input values
## Returns:
Tensor with type `C`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp2()?;
    println!("{}", b);
    Ok(())
}
```