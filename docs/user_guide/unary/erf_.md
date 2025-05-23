# erf_
```rust
erf_(x: &Tensor<T>, out: &mut Tensor<C> | Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{common::TensorInfo, error::TensorError, ops::FloatUnaryOps, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.erf_(&mut a.clone())?;
    println!("{}", b);
    assert_eq!(a.ptr().ptr as u64, b.ptr().ptr as u64);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |