# beta
```rust
beta(
    alpha: T,
    beta: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a beta distribution with parameters `alpha` and `beta`. The beta distribution is a continuous probability distribution defined on the interval [0, 1].
## Parameters:
`alpha`: Shape parameter alpha (α) of the beta distribution. Must be positive.

`beta`: Shape parameter beta (β) of the beta distribution. Must be positive.

`shape`: shape of the output
## Returns:
Tensor with type `T`
## Examples:
```rust
use hpt::{error::TensorError, ops::Random, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::beta(2.0, 5.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |