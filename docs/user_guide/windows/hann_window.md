# hann_window
```rust
hann_window(
    window_length: i64,
    periodic: bool
) -> Result<Tensor<T>, TensorError>
```
Creates a Hann window tensor. The Hann window is a taper formed by using a raised cosine with α = β = 0.5.

## Parameters:
`window_length`: The length of the window
`periodic`: If true, returns a window to be used as periodic function. If false, returns a symmetric window

## Returns:
A 1-D tensor containing the window.

## Examples:
```rust
use hpt_core::{Tensor, TensorError, WindowOps};
fn main() -> Result<(), TensorError> {
    // Create a periodic Hann window of length 5
    let a = Tensor::<f32>::hann_window(5, true)?;
    println!("{}", a);
    // [0.0000, 0.3455, 0.9045, 0.9045, 0.0000]

    // Create a symmetric Hann window of length 5
    let b = Tensor::<f32>::hann_window(5, false)?;
    println!("{}", b);
    // [0.0000, 0.5000, 1.0000, 0.5000, 0.0000]

    Ok(())
}
```