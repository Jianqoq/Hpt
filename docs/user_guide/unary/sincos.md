# sincos
```rust
sincos(x: &Tensor<T>) -> Result<(Tensor<C>, Tensor<C>), TensorError>
```
Simultaneously computes sine and cosine of the input tensor

## Parameters:
`x`: Angle(radians)

## Returns:
Tuple of two tensors (sine, cosine) with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let (sin, cos) = a.sincos()?;
    println!("sin: {}, cos: {}", sin, cos);
    Ok(())
}
```