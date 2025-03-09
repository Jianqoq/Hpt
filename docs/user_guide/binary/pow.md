# pow
```rust
pow(
    x: &Tensor<A> | Tensor<A>, 
    y: &Tensor<B> | Tensor<B> | scalar
) -> Result<Tensor<C>, TensorError>
```
Compute $\large x ^ y$ for all elements

## Parameters:
`x`: First input tensor

`y`: Second input tensor

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{error::TensorError, ops::FloatBinOps, Tensor};

fn main() -> Result<(), TensorError> {
    // Create tensors
    let a = Tensor::<f32>::new(&[2.0, 3.0, 4.0]);
    let b = Tensor::<f32>::new(&[2.0, 3.0, 2.0]);

    // Compute element-wise power
    let c = a.pow(&b)?;
    println!("{}", c);
    // Output:
    // [4.0, 27.0, 16.0]

    // Can also use scalar power
    let d = a.pow(2.0f64)?;
    println!("{}", d);
    // Output:
    // [4.0, 9.0, 16.0]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |