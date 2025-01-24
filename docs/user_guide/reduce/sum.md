# sum
```rust
sum(
    x: &Tensor<T>, 
    dims: 
        &[i64]
        | &[i64; len]
        | [i64; len] 
        | Vec<i64> 
        | &Vec<i64>
        | i64, 
    keepdim: bool
) -> Result<Tensor<T>, TensorError>
```
Compute the sum of elements along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{NormalReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Sum over dimension 0
    let a = Tensor::<f32>::new([10.0]);
    let b = a.sum([0], false)?;
    println!("{}", b); // [10.]

    // Sum over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.sum([0, 1], true)?;
    println!("{}", d); // [[10.]]
    Ok(())
}
```