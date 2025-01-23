# gelu
```rust
Tensor::<T>::gelu(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large x \cdot \Phi(x)$ where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.gelu()?;
    println!("{}", b);  // prints: 1.9545977
    Ok(())
}
```