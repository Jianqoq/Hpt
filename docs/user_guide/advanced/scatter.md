# scatter
```rust
scatter(
    x: &Tensor<T>,
    indices: &Tensor<I>,
    axis: i64,
    src: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Writes values from `src` tensor into a new tensor at the indices specified by `indices` along dimension `axis`. The rest of the values in the output tensor are copied from the input tensor `x`.

## Parameters:
`x`: Input tensor that provides the base values.

`indices`: Index tensor that specifies where to scatter the values.

`axis`: The axis along which to scatter values. Supports negative indexing.

`src`: The tensor containing values to scatter.

## Returns:
A new tensor with scattered values.

## Examples:
```rust
use hpt_core::{Tensor, TensorCreator, TensorError, AdvancedOps};

fn main() -> Result<(), TensorError> {
    let x = Tensor::<f64>::zeros(&[3, 5])?;  // base tensor
    let src = Tensor::<f64>::new(&[1., 2., 3.]);
    let indices = Tensor::<i64>::new(&[0, 2, 4]);
    
    let result = x.scatter(&indices, 1, &src)?;
    println!("After scatter:\n{}", result);
    Ok(())
}
```