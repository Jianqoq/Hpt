# nanprod
```rust
nanprod(
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
Compute the product of elements along the specified dimensions, treating NaNs as one

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
    // NaN product over dimension 0
    let a = Tensor::<f32>::new([2.0, f32::NAN, 4.0]);
    let b = a.nanprod([0], false)?;
    println!("{}", b); // [8.]

    // NaN product over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, f32::NAN], [3.0, 4.0]]);
    let d = c.nanprod([0, 1], true)?;
    println!("{}", d); // [[12.]]
    Ok(())
}
```