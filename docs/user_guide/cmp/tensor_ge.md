# tensor_ge
```rust
tensor_ge(
    x: &Tensor<A> | Tensor<A>, 
    y: &Tensor<B> | Tensor<B>
) -> Result<Tensor<bool>, TensorError>
```
check if element from x is greater or equal to the element from y

## Parameters:
`x`: First input tensor

`y`: Second input tensor

## Returns:
Tensor with type `bool`

## Examples:
```rust
use tensor_dyn::{TensorCmp, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    let b = a.tensor_ge(&a)?;
    println!("{}", b); // [true true true]
    Ok(())
}
```