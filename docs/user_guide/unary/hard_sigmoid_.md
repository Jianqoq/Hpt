# hard_sigmoid_
```rust
hard_sigmoid_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{max}(0, \text{min}(1, \frac{x}{6} + 0.5))$ for all elements with out. A piece-wise linear approximation of the sigmoid function.

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.hard_sigmoid_(&a)?;
    println!("{}", b);  // prints: 0.8333333
    Ok(())
}
```