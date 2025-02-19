# sum_square
```rust
sum_square(
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
Compute the sum of squares of elements along the specified dimensions

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
    // Sum of squares over dimension 0
    let a = Tensor::<f32>::new([2.0, 3.0]);
    let b = a.sum_square([0], false)?;
    println!("{}", b); // [13.]  // 2^2 + 3^2 = 13

    // Sum of squares over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.sum_square([0, 1], true)?;
    println!("{}", d); // [[30.]]  // 1^2 + 2^2 + 3^2 + 4^2 = 30
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |