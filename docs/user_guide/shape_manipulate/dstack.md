# dstack
```rust
dstack(
    tensors: Vec<&Tensor<T>>
) -> Result<Tensor<T>, TensorError>
```
Stacks tensors along the third axis (depth). Input tensors are promoted to 3D if necessary.

## Parameters:
`tensors`: Vector of tensors to stack along depth

## Returns:
A new tensor with input tensors stacked along the third axis.

## Examples:
```rust
use hpt::{
    error::TensorError,
    ops::{Concat, ShapeManipulate},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    // With 3D tensors
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2, 1])?;
    // [[[1], [2]],
    //  [[3], [4]]]
    let b = Tensor::<f32>::new(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2, 1])?;
    // [[[5], [6]],
    //  [[7], [8]]]

    let c = Tensor::dstack(vec![a.clone(), b.clone()])?;
    // [[[1, 5], [2, 6]],
    //  [[3, 7], [4, 8]]]
    println!("{}", c);

    // With 2D tensors (automatically adds depth dimension)
    let d = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    // [[1, 2],
    //  [3, 4]]
    let e = Tensor::dstack(vec![d.clone(), d.clone()])?;
    // [[[1, 1], [2, 2]],
    //  [[3, 3], [4, 4]]]
    println!("{}", e);

    // With 1D tensors (promoted to [1, n, 1])
    let f = Tensor::<f32>::new(&[1.0, 2.0]);
    let g = Tensor::<f32>::new(&[3.0, 4.0]);
    let h = Tensor::dstack(vec![f.clone(), g.clone()])?;
    // [[[1, 3], [2, 4]]]
    println!("{}", h);

    // With scalars (promoted to [1, 1, 1])
    let i = Tensor::<f32>::new(&[1.0]);
    let j = Tensor::dstack(vec![i.clone(), i.clone()])?;
    // [[[1, 1]]]
    println!("{}", j);

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |