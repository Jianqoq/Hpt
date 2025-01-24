# exp2_
```rust
exp2_(
    x: &Tensor<T>, 
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large 2^x$ for all elements with out
## Parameters:
`x`: Input values
`out`: Tensor to write to
## Returns:
Tensor with type `C`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp2_(&a)?;
    println!("{}", b);
    Ok(())
}
```