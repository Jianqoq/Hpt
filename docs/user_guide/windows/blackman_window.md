# blackman_window
```rust
blackman_window(
    window_length: i64,
    periodic: bool
) -> Result<Tensor<T>, TensorError>
```
Creates a Blackman window tensor. The Blackman window is a taper formed by using a weighted sum of cosine terms.

## Parameters:
`window_length`: The length of the window

`periodic`: If true, returns a window to be used as periodic function. If false, returns a symmetric window

## Returns:
A 1-D tensor containing the window.

## Examples:
```rust
use hpt_core::{Tensor, TensorError, WindowOps};
fn main() -> Result<(), TensorError> {
    // Create a periodic Blackman window of length 5
    let a = Tensor::<f32>::blackman_window(5, true)?;
    println!("{}", a);
    // [-0.0000, 0.2008, 0.8492, 0.8492, 0.2008]

    // Create a symmetric Blackman window of length 5
    let b = Tensor::<f32>::blackman_window(5, false)?;
    println!("{}", b);
    // [-0.0000, 0.3400, 1.0000, 0.3400]

    Ok(())
}
```