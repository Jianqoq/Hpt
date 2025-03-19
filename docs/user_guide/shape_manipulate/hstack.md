# hstack
```rust
hstack(
    tensors: Vec<Tensor<T>>
) -> Result<Tensor<T>, TensorError>
```
Stacks tensors horizontally (along axis 1 for 2D+ tensors, or axis 0 for 1D tensors).

## Parameters:
`tensors`: Vector of tensors to stack horizontally

## Returns:
A new tensor with input tensors stacked horizontally.

## Examples:
```rust
use hpt::{
    error::TensorError,
    ops::{Concat, ShapeManipulate},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    // With 2D tensors
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    // [[1, 2],
    //  [3, 4]]
    let b = Tensor::<f32>::new(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2])?;
    // [[5, 6],
    //  [7, 8]]

    let c = Tensor::hstack(vec![a.clone(), b.clone()])?;
    // [[1, 2, 5, 6],
    //  [3, 4, 7, 8]]
    println!("{}", c);

    // With 1D tensors
    let d = Tensor::<f32>::new(&[1.0, 2.0]);
    let e = Tensor::<f32>::new(&[3.0, 4.0]);
    let f = Tensor::hstack(vec![d.clone(), e.clone()])?;
    // [1, 2, 3, 4]
    println!("{}", f);

    // With scalars (0D tensors)
    let g = Tensor::<f32>::new(&[1.0]);
    let h = Tensor::<f32>::new(&[2.0]);
    let i = Tensor::hstack(vec![g.clone(), h.clone()])?;
    // [1, 2]
    println!("{}", i);

    // Will raise an error if heights don't match for 2D tensors
    let j = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3, 1])?;
    assert!(Tensor::hstack(vec![a.clone(), j.clone()]).is_err());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |