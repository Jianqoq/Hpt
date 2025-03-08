# tensor_where
```rust
tensor_where(
    condition: &Tensor<bool>,
    x: &Tensor<T>,
    y: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Element-wise selection based on a condition tensor. Returns a tensor of elements selected from `x` where condition is true, and from `y` where condition is false.

## Parameters:
`condition`: A boolean tensor that determines which elements to select.

`x`: Tensor whose elements are selected where condition is true.

`y`: Tensor whose elements are selected where condition is false.

## Returns:
A new tensor with elements selected from `x` and `y` based on the condition.

## Examples:
```rust
use hpt::{error::TensorError, ops::TensorWhere, Tensor};

fn main() -> Result<(), TensorError> {
    let condition = Tensor::<bool>::new(&[true, false, true]);
    let x = Tensor::<f64>::new(&[1., 2., 3.]);
    let y = Tensor::<f64>::new(&[4., 5., 6.]);

    let result = Tensor::tensor_where(&condition, &x, &y)?;
    println!("{}", result); // [1., 5., 3.]

    Ok(())
}
```