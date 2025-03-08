# arange
```rust
arange(
    start: T,
    end: T
) -> Result<Tensor<T>, TensorError>
```
Creates a 1-D tensor with evenly spaced values within a given interval `[start, end)`.

## Parameters:
`start`: Start of interval (inclusive)

`end`: End of interval (exclusive)

## Returns:
A 1-D tensor with values from `start` to `end-1`.

## Examples:
```rust
use hpt::{Tensor, error::TensorError, ops::TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create sequence from 0 to 5
    let a = Tensor::<f32>::arange(0, 5)?;
    println!("{}", a);
    // [0, 1, 2, 3, 4]

    // Using floating point numbers
    let b = Tensor::<f32>::arange(1.5, 5.5)?;
    println!("{}", b);
    // [1.5, 2.5, 3.5, 4.5]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |