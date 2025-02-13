# arange_step
```rust
arange_step(
    start: T,
    end: T,
    step: T
) -> Result<Tensor<T>, TensorError>
```
Creates a 1-D tensor with evenly spaced values within a given interval `[start, end)` with a specified step size.

## Parameters:
`start`: Start of interval (inclusive)

`end`: End of interval (exclusive)

`step`: Size of spacing between values

## Returns:
A 1-D tensor with values from `start` to `end-1` with step size `step`.

## Examples:
```rust
use hpt::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create sequence with step 2
    let a = Tensor::<f32>::arange_step(0.0, 5.0, 2.0)?;
    println!("{}", a);
    // [0, 2, 4]

    // Using negative step
    let b = Tensor::<f32>::arange_step(5.0, 0.0, -1.0)?;
    println!("{}", b);
    // [5, 4, 3, 2, 1]

    Ok(())
}
```