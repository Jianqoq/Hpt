# logsumexp
```rust
logsumexp(
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
Compute $\log(\sum_{i} \exp(x_i))$ along the specified dimensions. This function is numerically more stable than computing the log of sum of exponentials directly.

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
    // LogSumExp over dimension 0
    let a = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    let b = a.logsumexp([0], false)?;
    println!("{}", b); // [3.4076061]

    // LogSumExp over multiple dimensions with keepdim=true
    let c = Tensor::<f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let d = c.logsumexp([0, 1], true)?;
    println!("{}", d); // [[4.4401898]]
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |