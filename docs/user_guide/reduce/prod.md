# prod
```rust
prod(
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
Compute the product of elements along the specified dimensions

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
    // Product over dimension 0
    let a = Tensor::<f32>::new([2.0, 3.0, 4.0]);
    let b = a.prod([0], false)?;
    println!("{}", b); // [24.]

    // Product over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.prod([0, 1], true)?;
    println!("{}", d); // [[24.]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |