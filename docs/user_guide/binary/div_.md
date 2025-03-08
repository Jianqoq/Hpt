# div_
```rust
div_(
    x: &Tensor<A> | Tensor<A> | scalar, 
    y: &Tensor<B> | Tensor<B> | scalar,
    out: &mut Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large x / y$ for all elements with out

## Parameters:
`x`: First input tensor

`y`: Second input tensor

`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatBinOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = Tensor::<f32>::new([3.0]);
    let c = a.div_(&b, &mut a.clone())?;
    println!("{}", c);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |