# reducel3
```rust
reducel3(
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
Compute the L3 norm $(\sum_{i} |x_i|^3)^{\frac{1}{3}}$ along the specified dimensions

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
    // L3 norm over dimension 0
    let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    let b = a.reducel3([0], false)?;
    println!("{}", b); // [2.8845]

    // L3 norm over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [2.0, 3.0]]);
    let d = c.reducel3([0, 1], true)?;
    println!("{}", d); // [[3.5303]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |