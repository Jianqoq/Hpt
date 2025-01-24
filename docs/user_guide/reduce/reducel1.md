# reducel1
```rust
reducel1(
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
Compute the L1 norm (sum of absolute values) $\sum_{i} |x_i|$ along the specified dimensions

## Parameters:
`x`: Input tensor

`dims`: Dimensions to reduce over

`keepdim`: Whether to keep the reduced dimensions with length 1

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatReduce, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // L1 norm over dimension 0
    let a = Tensor::<f32>::new([-1.0, 2.0, -3.0]);
    let b = a.reducel1([0], false)?;
    println!("{}", b); // [6.]  // |-1| + |2| + |-3| = 6

    // L1 norm over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[-1.0, 2.0], [-3.0, 4.0]]);
    let d = c.reducel1([0, 1], true)?;
    println!("{}", d); // [[10.]]  // |-1| + |2| + |-3| + |4| = 10
    Ok(())
}
```