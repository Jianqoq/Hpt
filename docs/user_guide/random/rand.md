# rand
```rust
rand(
    shape: &[i64] | &Vec<i64> | &[i64; _], 
    low: T, 
    high: T
) -> Result<Tensor<T>, TensorError>
```
create a Tensor with data uniformly distributed between `low` and `high`.
## Parameters:
`shape`: shape of the output

`low`: the lowest value

`high`: the highest value
## Returns:
Tensor with type `T`
## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::rand([10, 10], 0.0, 10.0)?;
    println!("{}", a);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |