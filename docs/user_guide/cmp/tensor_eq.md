# tensor_eq
```rust
tensor_eq(
    x: &Tensor<A> | Tensor<A>, 
    y: &Tensor<B> | Tensor<B>
) -> Result<Tensor<bool>, TensorError>
```
check if element from x is equal to element from y

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
    let b = a.tensor_eq(&a)?;
    println!("{}", b); // [true true true]
    Ok(())
}
```