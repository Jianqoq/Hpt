# hard_swish_
```rust
hard_swish_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large x \cdot \text{min}(\text{max}(0, \frac{x}{6} + 0.5), 1)$ for all elements with out. A piece-wise linear approximation of the swish function.

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.hard_swish_(&a)?;
    println!("{}", b);  // prints: 1.6666666
    Ok(())
}
```