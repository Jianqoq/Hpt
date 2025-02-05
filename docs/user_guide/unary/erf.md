# erf
```rust
erf(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([1.0]);
    let b = a.erf()?;
    println!("{}", b);  // prints: 0.8427008
    Ok(())
}
```