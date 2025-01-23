# softsign
```rust
Tensor::<T>::softsign(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \frac{x}{1 + |x|}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.softsign()?;
    println!("{}", b);  // prints: 0.6666667
    Ok(())
}
```