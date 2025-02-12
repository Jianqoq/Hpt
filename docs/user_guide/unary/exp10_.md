# exp10_
```rust
exp10_(
    x: &Tensor<T>, 
    out: &mut Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute 10 raised to the power of `x` for all elements with output tensor

## Parameters:
`x`: Input values (exponents)  
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);  // 10^2
    let b = a.exp10_(&mut a.clone())?;
    println!("{}", b);  // prints: 100.0
    Ok(())
}
```