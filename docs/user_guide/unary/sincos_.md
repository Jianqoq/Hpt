# sincos_
```rust
sincos_(
    x: &Tensor<T>, 
    out: (Tensor<C> | &mut Tensor<C>, Tensor<C> | &mut Tensor<C>)
) -> Result<(Tensor<C>, Tensor<C>), TensorError>
```
Simultaneously computes sine and cosine of the input tensor with output tensors

## Parameters:
`x`: Angle(radians)  
`out`: Tuple of two tensors to write results to (sine, cosine)

## Returns:
Tuple of two tensors (sine, cosine) with type `C`

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{FloatUnaryOps, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let mut sin_out = Tensor::<f32>::zeros_like(&a)?;
    let mut cos_out = Tensor::<f32>::zeros_like(&a)?;
    let (sin, cos) = a.sincos_((&mut sin_out, &mut cos_out))?;
    println!("sin: {}, cos: {}", sin, cos);
    assert_eq!(sin.ptr().ptr as u64, sin_out.ptr().ptr as u64);
    assert_eq!(cos.ptr().ptr as u64, cos_out.ptr().ptr as u64);
    Ok(())
}

```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |