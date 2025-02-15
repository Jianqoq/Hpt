# cumsum
```rust
fn cumsum<A: Into<Option<i64>>>(
    &self,
    axis: A
) -> Result<Tensor<T>, TensorError>
```
Computes the cumulative sum of tensor elements along a specified axis.

## Parameters:
`axis`: The axis along which to compute the cumulative sum
- If specified: Compute cumsum along this axis
- If None: Compute cumsum over the flattened array

## Returns:
A new tensor of the same shape containing the cumulative sum.

## Examples:
```rust
use hpt::{CumulativeOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // 1D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]);
    let cum_a = a.cumsum(None)?;
    println!("{}", cum_a);
    // [1.0, 3.0, 6.0]

    // 2D tensor
    let b = Tensor::<f32>::new(&[[1.0, 2.0], [3.0, 4.0]]);

    // Cumsum along axis 0
    let cum_b0 = b.cumsum(0)?;
    println!("{}", cum_b0);
    // [[1.0, 2.0],
    //  [4.0, 6.0]]

    // Cumsum along axis 1
    let cum_b1 = b.cumsum(1)?;
    println!("{}", cum_b1);
    // [[1.0, 3.0],
    //  [3.0, 7.0]]

    Ok(())
}
```