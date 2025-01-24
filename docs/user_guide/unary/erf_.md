# erf_
```rust
erf_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([1.0]);
    let b = a.erf_(&a)?;
    println!("{}", b);  // prints: 0.8427008
    Ok(())
}
```