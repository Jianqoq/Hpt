# swap_axes
```rust
swap_axes(
    x: &Tensor<T>,
    axis1: i64,
    axis2: i64
) -> Result<Tensor<T>, TensorError>
```
Interchanges two axes of a tensor. This operation creates a view of the tensor with the specified axes swapped.

## Parameters:
`x`: Input tensor

`axis1`: First axis to be swapped

`axis2`: Second axis to be swapped

Both axes can be negative, counting from the end of the dimensions.

## Returns:
A new tensor with the specified axes swapped.

## Examples:
```rust
use hpt_core::{ShapeManipulate, Tensor, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor with shape [2, 3]
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Swap axes 0 and 1
    let b = a.swap_axes(0, 1)?;
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    println!("{}", b);

    // Create a 3D tensor with shape [2, 3, 2]
    let c = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                                .reshape(&[2, 3, 2])?;

    // Using negative indices (-1 means last axis)
    let d = c.swap_axes(0, -1)?;
    println!("{}", d.shape()); // prints [2, 3, 2]

    // Will raise an error for invalid axis
    assert!(a.swap_axes(0, 2).is_err());

    Ok(())
}
```