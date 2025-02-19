# all
```rust
all(
    x: &Tensor<T>, 
    dims: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64, 
    keepdim: bool
) -> Result<Tensor<bool>, TensorError>
```
Test if all elements are true along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `bool`

## Examples:
```rust
use hpt::{EvalReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Max over dimension 0
    let a = Tensor::<f32>::new([10.0]);
    let b = a.all([0], false)?;
    println!("{}", b); // [true]

    // Max over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.all([0, 1], true)?;
    println!("{}", d); // [[true]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |