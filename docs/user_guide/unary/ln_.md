# ln_
```rust
ln_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \ln(x)$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.718281828459045]);
    let b = a.ln_(&a)?;
    println!("{}", b);  // prints: 1.0
    Ok(())
}
```