# mish_
```rust
mish_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large x * \tanh(\ln(1 + e^x))$ for all elements with out

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
    let b = a.mish_(&a)?;
    println!("{}", b);
    Ok(())
}
```