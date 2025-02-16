# bernoulli
```rust
bernoulli(
    shape: &[i64] | &Vec<i64> | &[i64; _],
    p: T
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a Bernoulli distribution. The Bernoulli distribution is a discrete probability distribution of a random variable which takes the value 1 with probability p and the value 0 with probability q = 1 - p.

## Parameters:
`shape`: Shape of the output tensor.

`p`: Success probability (p). Must be in the interval [0, 1].

## Returns:
Tensor with type `T` containing random values (0 or 1) from the Bernoulli distribution.

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with Bernoulli distribution (p=0.7)
    let a = Tensor::<f32>::bernoulli(&[10, 10], 0.7)?;
    println!("{}", a);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |