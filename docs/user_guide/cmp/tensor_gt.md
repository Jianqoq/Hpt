# tensor_gt
```rust
tensor_gt(
    x: &Tensor<A> | Tensor<A>, 
    y: &Tensor<B> | Tensor<B>
) -> Result<Tensor<bool>, TensorError>
```
check if element from x is greater than the element from y

## Parameters:
`x`: First input tensor

`y`: Second input tensor

## Returns:
Tensor with type `bool`

## Examples:
```rust
use hpt::{ops::TensorCmp, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    let b = a.tensor_gt(&a)?;
    println!("{}", b); // [false false false]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |