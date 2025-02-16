# reducel2
```rust
reducel2(
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
Compute the L2 norm (Euclidean norm) $\sqrt{\sum_{i} x_i^2}$ along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // L2 norm over dimension 0
    let a = Tensor::<f32>::new([3.0, 4.0]);
    let b = a.reducel2([0], false)?;
    println!("{}", b); // [5.]  // sqrt(3^2 + 4^2) = 5

    // L2 norm over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [2.0, 3.0]]);
    let d = c.reducel2([0, 1], true)?;
    println!("{}", d); // [[4.2426405]]  // sqrt(1^2 + 2^2 + 2^2 + 3^2) ≈ 4.24
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |