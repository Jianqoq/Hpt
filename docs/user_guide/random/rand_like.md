# rand_like
```rust
rand_like(
    x: &Tensor<T>, 
    low: T, 
    high: T
) -> Result<Tensor<T>, TensorError>
```
same as `rand` but the shape will be based on `x`.
## Parameters:
`x`: input Tensor

`low`: the lowest value

`high`: the highest value
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::rand([10, 10], 0.0, 10.0)?;
    println!("{}", a.rand_like(0.0, 10.0)?);
    Ok(())
}
```