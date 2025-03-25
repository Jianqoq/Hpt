# allclose
```rust
fn astype<NewType>(x: &Tensor<T>) -> Result<Tensor<NewType>, TensorError>
```
Casts a tensor to a different data type.

## Parameters:
`x`: The input tensor to be cast

`NewType`: The target data type (specified through type parameter in the function call)

## Returns:
A new tensor with the same shape as the input tensor but with elements cast to the type U, or a TensorError if the cast operation fails.

## Examples:
```rust
use hpt::{Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    // Create a tensor with f32 values
    let a = Tensor::<f32>::new([1.5, 2.7, 3.2]);
    
    // Cast to i32 (floating-point values will be truncated)
    let a_int = a.astype::<i32>()?;
    println!("{}", a_int); // [1 2 3]
    
    // Cast to f64 (higher precision)
    let a_double = a.astype::<f64>()?;
    println!("{}", a_double); // [1.5 2.7 3.2]
    
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |