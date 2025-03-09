# hypot
```rust
hypot(
    x: &Tensor<A> | Tensor<A>, 
    y: &Tensor<B> | Tensor<B> | scalar
) -> Result<Tensor<C>, TensorError>
```
Compute $\large \sqrt{x^2 + y^2}$ for all elements

## Parameters:
`x`: First input tensor

`y`: Second input tensor or scalar

## Returns:
Tensor with type `C`, containing the hypotenuse values

## Examples:
```rust
use hpt::{error::TensorError, ops::FloatBinOps, Tensor};

fn main() -> Result<(), TensorError> {
    // Create tensors representing the base and height of triangles
    let base = Tensor::<f32>::new(&[3.0, 4.0, 5.0]);
    let height = Tensor::<f32>::new(&[4.0, 3.0, 12.0]);

    // Compute hypotenuse lengths
    let hypotenuse = base.hypot(&height)?;
    println!("{}", hypotenuse);
    // Output:
    // [5.0, 5.0, 13.0]

    // Can also use with scalar
    let fixed_height = base.hypot(4.0)?;
    println!("{}", fixed_height);
    // Output:
    // [5.0, 5.66, 6.40]

    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅        |
| Cuda    | ✅        |