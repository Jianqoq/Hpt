# mul_
```rust
mul_(
    x: Tensor<A>, 
    y: &Tensor<B> | Tensor<B> | scalar,
    out: &mut Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large x * y$ for all elements with out

## Parameters:
`x`: First input tensor

`y`: Second input tensor

`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{error::TensorError, ops::NormalBinOps, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = Tensor::<f32>::new([3.0]);
    let c = a.mul_(&b, &mut a.clone())?;
    println!("{}", c);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |