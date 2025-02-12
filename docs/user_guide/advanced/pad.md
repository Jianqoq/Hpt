# pad
```rust
pad(
    x: &Tensor<T>,
    pads: &[(i64, i64)],
    val: T
) -> Result<Tensor<T>, TensorError>
```
Pads a tensor with a given constant value. For each dimension, adds padding at the start and end as specified by `pads`.

## Parameters:
`x`: Input tensor to be padded.

`pads`: A slice of tuples where each tuple contains two values (before_pad, after_pad) for each dimension. The length must match the number of dimensions in the input tensor.

`val`: The constant value to use for padding.

## Returns:
A new tensor with padding applied.

## Examples:
```rust
use hpt_core::{AdvancedOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let x = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]); // 2x2 matrix

    // Pad with zeros:
    // - Add 1 row before, 0 rows after
    // - Add 0 columns before, 2 columns after
    let pads = &[(1, 0), (0, 2)];
    let result = x.pad(pads, 0.0)?;
    println!("After padding:\n{}", result);
    // Output:
    // [[0., 0., 0., 0.]
    //  [1., 2., 0., 0.]
    //  [3., 4., 0., 0.]]

    Ok(())
}
```