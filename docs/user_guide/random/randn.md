# randn
```rust
randn(shape: &[i64] | &Vec<i64> | &[i64; _]) -> Result<Tensor<T>, TensorError>
```
create a Tensor with data in normal distribution. `mean = 0.0`, `std_dev = 1.0`.
## Parameters:
`shape`: shape of the output
## Returns:
Tensor with type `T`
## Examples:
```rust
use hpt_core::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::randn([10, 10])?;
    println!("{}", a);
    Ok(())
}
```