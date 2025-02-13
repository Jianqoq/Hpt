# add_
```rust
add_(
    x: &Tensor<A> | Tensor<A> | scalar, 
    y: &Tensor<B> | Tensor<B> | scalar,
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large x + y$ for all elements with out

## Parameters:
`x`: First input tensor

`y`: Second input tensor

`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatBinaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = Tensor::<f32>::new([3.0]);
    let c = a.add_(&b, &a)?;
    println!("{}", c);
    Ok(())
}
```