# gelu_
```rust
gelu_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large x \cdot \Phi(x)$ where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution for all elements with out

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
    let b = a.gelu_(&a)?;
    println!("{}", b);  // prints: 1.9545977
    Ok(())
}
```