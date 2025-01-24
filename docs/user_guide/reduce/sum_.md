# sum_
```rust
sum_(
    x: &Tensor<T>, 
    dims: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64,
    keepdim: bool,
    init_out: bool,
    out: &Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Compute the sum of elements along the specified dimensions with out

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

`init_out`: init the out data before doing computation   

`out`: Tensor to write to

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{NormalReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Sum over dimension 0
    let a = Tensor::<f32>::new([10.0]);
    let b = Tensor::<f32>::new([0.0]);
    let c = a.sum_([0], false, false, &b)?;
    println!("{}", c); // [10.]

    // Sum over multiple dimensions with keepdim=true
    let d = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let e = Tensor::<f32>::new([[0.0]]);
    let f = d.sum_([0, 1], true, false, &e)?;
    println!("{}", f); // [[10.]]
    Ok(())
}
```