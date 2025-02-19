# nansum
```rust
nansum(
    x: &Tensor<T>, 
    dims: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64, 
    keepdim: bool
) -> Result<Tensor<T>, TensorError>
```
Compute the sum of elements along the specified dimensions, treating NaNs as zero

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{NormalReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // NaN sum over dimension 0
    let a = Tensor::<f32>::new([1.0, f32::NAN, 3.0]);
    let b = a.nansum([0], false)?;
    println!("{}", b); // [4.]

    // NaN sum over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, f32::NAN], [3.0, 4.0]]);
    let d = c.nansum([0, 1], true)?;
    println!("{}", d); // [[8.]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |