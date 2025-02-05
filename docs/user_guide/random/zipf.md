# zipf
```rust
zipf(
    n: u64,
    s: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a Zipf distribution. The Zipf distribution is a discrete probability distribution commonly used to model frequency distributions of ranked data in various physical and social phenomena, where the frequency of any element is inversely proportional to its rank.

## Parameters:
`n`: Number of elements (N). Defines the range of possible values [1, N].

`s`: Exponent parameter (s). Controls the skewness of the distribution. Must be greater than 1.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the Zipf distribution.

## Examples:
```rust
use hpt_core::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with Zipf distribution (N=1000, s=2.0)
    let a = Tensor::<f32>::zipf(1000, 2.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```