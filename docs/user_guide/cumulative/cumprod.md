# cumprod
```rust
fn cumprod<A: Into<Option<i64>>>(
    &self,
    axis: A
) -> Result<Tensor<T>, TensorError>
```
Computes the cumulative product of tensor elements along a specified axis.

## Parameters:
`axis`: The axis along which to compute the cumulative product
- If specified: Compute cumprod along this axis
- If None: Compute cumprod over the flattened array

## Returns:
A new tensor of the same shape containing the cumulative product.

## Examples:
```rust
use hpt::{error::TensorError, ops::CumulativeOps, Tensor};

fn main() -> Result<(), TensorError> {
    // 1D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]);
    let cum_a = a.cumprod(None)?;
    println!("{}", cum_a);
    // [1.0, 2.0, 6.0]

    // 2D tensor
    let b = Tensor::<f32>::new(&[[1.0, 2.0], [3.0, 4.0]]);

    // Cumprod along axis 0
    let cum_b0 = b.cumprod(0)?;
    println!("{}", cum_b0);
    // [[1.0, 2.0],
    //  [3.0, 8.0]]

    // Cumprod along axis 1
    let cum_b1 = b.cumprod(1)?;
    println!("{}", cum_b1);
    // [[1.0, 2.0],
    //  [3.0, 12.0]]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |