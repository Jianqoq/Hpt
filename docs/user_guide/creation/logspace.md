# logspace
```rust
logspace(
    start: T,
    end: T,
    num: usize,
    include_end: bool,
    base: T
) -> Result<Tensor<T>, TensorError>
where
    T: Float + FromScalar<f64>
```
Creates a 1-D tensor with `num` numbers logarithmically spaced between `base^start` and `base^end`.

## Parameters:
`start`: The starting value of the sequence (power of base)
`end`: The end value of the sequence (power of base)
`num`: Number of samples to generate
`include_end`: Whether to include the end value in the sequence
`base`: The base of the log space (default is 10.0)

## Returns:
A 1-D tensor of logarithmically spaced values.

## Examples:
```rust
use hpt_core::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create 4 points from 10^0 to 10^3
    let a = Tensor::<f32>::logspace(0.0, 3.0, 4, true, 10.0)?;
    println!("{}", a);
    // [1.0, 10.0, 100.0, 1000.0]

    // Using base 2
    let b = Tensor::<f32>::logspace(0.0, 3.0, 4, true, 2.0)?;
    println!("{}", b);
    // [1.0, 2.0, 4.0, 8.0]

    // Excluding end point
    let c = Tensor::<f32>::logspace(0.0, 2.0, 4, false, 10.0)?;
    println!("{}", c);
    // [1.0, 3.1623, 10.0, 31.6228]

    Ok(())
}
```