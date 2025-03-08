# mean
```rust
mean(
    x: &Tensor<T>, 
    dims: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64, 
    keepdim: bool
) -> Result<Tensor<C>, TensorError>
```
Compute the mean of elements along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::NormalReduce, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    // Mean over dimension 0
    let a = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    let b = a.mean([0], false)?;
    println!("{}", b); // [2.]

    // Mean over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.mean([0, 1], true)?;
    println!("{}", d); // [[2.5]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |