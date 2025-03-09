# min
```rust
min(
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
Find the minimum of element along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{ops::NormalReduce, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    // Min over dimension 0
    let a = Tensor::<f32>::new([10.0]);
    let b = a.min([0], false)?;
    println!("{}", b); // [10.]

    // Min over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.min([0, 1], true)?;
    println!("{}", d); // [[1.]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |