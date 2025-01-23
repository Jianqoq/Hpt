# hard_swish
```rust
Tensor::<T>::hard_swish(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large x \cdot \text{min}(\text{max}(0, \frac{x}{6} + 0.5), 1)$ for all elements. A piece-wise linear approximation of the swish function.

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.hard_swish()?;
    println!("{}", b);  // prints: 1.6666666
    Ok(())
}
```