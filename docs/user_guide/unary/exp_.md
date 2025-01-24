# exp_
```rust
exp_(
    x: &Tensor<T>, 
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute exponential of `x` for all elements with out
## Parameters:
`x`: Input values
`out`: Tensor to write to
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp_(&a)?;
    println!("{}", b);
    Ok(())
}
```