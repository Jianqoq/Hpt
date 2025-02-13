# hamming_window
```rust
hamming_window(
    window_length: i64,
    periodic: bool
) -> Result<Tensor<T>, TensorError>
```
Creates a Hamming window tensor. The Hamming window is a taper formed by using a weighted cosine.

## Parameters:
`window_length`: The length of the window

`periodic`: If true, returns a window to be used as periodic function. If false, returns a symmetric window

## Returns:
A 1-D tensor containing the window.

## Examples:
```rust
use hpt::{Tensor, TensorError, WindowOps};
fn main() -> Result<(), TensorError> {
    // Create a periodic Hamming window of length 5
    let a = Tensor::<f32>::hamming_window(5, true)?;
    println!("{}", a);
    // [0.0800, 0.3979, 0.9121, 0.9121, 0.0800]

    // Create a symmetric Hamming window of length 5
    let b = Tensor::<f32>::hamming_window(5, false)?;
    println!("{}", b);
    // [0.08, 0.54, 1.00, 0.54]

    Ok(())
}
```