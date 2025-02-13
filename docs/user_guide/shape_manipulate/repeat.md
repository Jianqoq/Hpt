# repeat
```rust
repeat(
    x: &Tensor<T>,
    repeats: usize,
    axis: i16
) -> Result<Tensor<T>, TensorError>
```
Repeats elements of a tensor along a specified axis.

## Parameters:
`x`: Input tensor

`repeats`: Number of repetitions for each element

`axis`: The axis along which to repeat values. Negative values count from the end

## Returns:
A new tensor with repeated elements along the specified axis.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    // [[1, 2],
    //  [3, 4]]

    // Repeat 2 times along axis 0 (rows)
    let b = a.repeat(2, 0)?;
    // [[1, 2],
    //  [1, 2],
    //  [3, 4],
    //  [3, 4]]
    println!("{}", b);

    // Repeat 3 times along axis 1 (columns)
    let c = a.repeat(3, 1)?;
    // [[1, 1, 1, 2, 2, 2],
    //  [3, 3, 3, 4, 4, 4]]
    println!("{}", c);

    // Using negative axis (-1 means last axis)
    let d = a.repeat(2, -1)?;
    // [[1, 1, 2, 2],
    //  [3, 3, 4, 4]]
    println!("{}", d);

    Ok(())
}
```