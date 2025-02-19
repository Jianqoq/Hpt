# tensor_lt
```rust
tensor_lt(
    x: &Tensor<A> | Tensor<A>, 
    y: &Tensor<B> | Tensor<B>
) -> Result<Tensor<bool>, TensorError>
```
check if element from x is less than the element from y

## Parameters:
`x`: First input tensor

`y`: Second input tensor

## Returns:
Tensor with type `bool`

## Examples:
```rust
use hpt::{TensorCmp, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    let b = a.tensor_lt(&a)?;
    println!("{}", b); // [true true true]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |