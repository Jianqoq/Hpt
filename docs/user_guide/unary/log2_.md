# log2_
```rust
Tensor::<T>::log2_(x: &Tensor<T>, out: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \log_{2}(x)$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([8.0]);
    let b = a.log2_(&a)?;
    println!("{}", b);  // prints: 3.0
    Ok(())
}
```