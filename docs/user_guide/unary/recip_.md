# recip_
```rust
Tensor::<T>::recip_(x: &Tensor<T>, out: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \frac{1}{x}$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.recip_(&a)?;
    println!("{}", b);
    Ok(())
}
```