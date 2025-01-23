# celu
```rust
Tensor::<T>::celu(x: &Tensor<T>, alpha: T) -> Result<Tensor<T>, TensorError>
```
Compute $\large \text{max}(0, x) + \text{min}(0, \alpha \cdot (e^{x/\alpha} - 1))$ for all elements

## Parameters:
`x`: Input values

`alpha`: Parameter controlling the saturation of negative values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-1.0]);
    let b = a.celu(1.0)?;
    println!("{}", b);  // prints: -0.6321206
    Ok(())
}
```