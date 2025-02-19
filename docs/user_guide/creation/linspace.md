# linspace
```rust
linspace(
    start: T,
    end: T,
    num: usize,
    include_end: bool
) -> Result<Tensor<T>, TensorError>
```
Creates a 1-D tensor of `num` evenly spaced values between `start` and `end`.

## Parameters:
`start`: The starting value of the sequence

`end`: The end value of the sequence

`num`: Number of samples to generate

`include_end`: Whether to include the end value in the sequence

## Returns:
A 1-D tensor of evenly spaced values.

## Examples:
```rust
use hpt::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create 5 evenly spaced values from 0 to 1 (inclusive)
    let a = Tensor::<f32>::linspace(0.0, 1.0, 5, true)?;
    println!("{}", a);
    // [0.0, 0.25, 0.5, 0.75, 1.0]

    // Create 5 evenly spaced values from 0 to 1 (exclusive)
    let b = Tensor::<f32>::linspace(0.0, 1.0, 5, false)?;
    println!("{}", b);
    // [0.0, 0.2, 0.4, 0.6, 0.8]

    // Using integer endpoints
    let c = Tensor::<f32>::linspace(0, 10, 6, true)?;
    println!("{}", c);
    // [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |