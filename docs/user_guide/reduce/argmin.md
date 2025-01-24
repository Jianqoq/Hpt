# argmin
```rust
argmin(
    x: &Tensor<T>, 
    dims: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64, 
    keepdim: bool
) -> Result<Tensor<i64>, TensorError>
```
Return the indices of the minimum values along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `i64` containing the indices of minimum values

## Examples:
```rust
use tensor_dyn::{IndexReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Max over dimension 0
    let a = Tensor::<f32>::new([10.0]);
    let b = a.argmin([0], false)?;
    println!("{}", b); // [0]
    Ok(())
}
```