# allclose
```rust
fn allclose(&self, other: &Tensor<T>, rtol: T, atol: T) -> bool
```
Checks whether two tensors are element-wise equal within a tolerance with the formula `|a - b| <= atol + rtol * |b|`

## Parameters:
`x`: First input tensor

`y`: Second input tensor

`rtol`: Relative tolerance - maximum allowed relative difference

`atol`: Absolute tolerance - maximum allowed absolute difference

## Returns:
`true` if all elements in both tensors are equal within the specified tolerance, otherwise `false`.

## Examples:
```rust
use hpt::{Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    let b = Tensor::<f32>::new([1.0001, 2.0001, 3.0001]);
    
    // With small tolerance - returns false
    let result1 = a.allclose(&b, 1e-5, 1e-5);
    println!("With small tolerance: {}", result1); // false
    
    // With larger tolerance - returns true
    let result2 = a.allclose(&b, 1e-3, 1e-3);
    println!("With larger tolerance: {}", result2); // true
    
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |