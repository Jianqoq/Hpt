# hypot_
```rust
hypot_(
    x: Tensor<A>, 
    y: &Tensor<B> | Tensor<B> | scalar,
    out: &mut Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large \sqrt{x^2 + y^2}$ for all elements with out

## Parameters:
`x`: First input tensor

`y`: Second input tensor or scalar

`out`: Tensor to write to

## Examples:
```rust
use hpt::{error::TensorError, ops::FloatBinOps, Tensor};

fn main() -> Result<(), TensorError> {
    // Create tensors representing the base and height of triangles
    let base = Tensor::<f32>::new(&[3.0, 4.0, 5.0]);
    let height = Tensor::<f32>::new(&[4.0, 3.0, 12.0]);

    // Compute hypotenuse lengths and store in base
    let result = base.hypot_(&height, &mut base.clone())?;
    println!("{}", result);
    // Output:
    // [5.0, 5.0, 13.0]

    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅        |
| Cuda    | ✅        |