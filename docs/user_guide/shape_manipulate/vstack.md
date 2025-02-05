# vstack
```rust
vstack(
    tensors: Vec<&Tensor<T>>
) -> Result<Tensor<T>, TensorError>
```
Stacks tensors vertically (along axis 0). This is equivalent to concatenation along the first axis.

## Parameters:
`tensors`: Vector of tensors to stack vertically

## Returns:
A new tensor with input tensors stacked vertically.

## Examples:
```rust
use hpt_core::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create two 2D tensors
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[1, 3])?;
    // [[1, 2, 3]]
    let b = Tensor::<f32>::new(&[4.0, 5.0, 6.0]).reshape(&[1, 3])?;
    // [[4, 5, 6]]

    // Stack vertically
    let c = Tensor::vstack(vec![&a, &b])?;
    // [[1, 2, 3],
    //  [4, 5, 6]]
    println!("{}", c);

    // Works with 1D tensors too
    let d = Tensor::<f32>::new(&[1.0, 2.0, 3.0]);
    let e = Tensor::<f32>::new(&[4.0, 5.0, 6.0]);
    let f = Tensor::vstack(vec![&d, &e])?;
    // [1, 2, 3, 4, 5, 6]
    println!("{}", f);

    // Will raise an error if widths don't match
    let g = Tensor::<f32>::new(&[1.0, 2.0]).reshape(&[1, 2])?;
    assert!(Tensor::vstack(vec![&a, &g]).is_err());

    Ok(())
}
```