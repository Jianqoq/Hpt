# elu_
```rust
elu_(x: &Tensor<T>, alpha: C, out: &mut Tensor<C> | Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large x$ for $x > 0$, $\large \alpha(e^x - 1)$ for $x \leq 0$ for all elements with out

## Parameters:
`x`: Input values
`alpha`: Parameter controlling the saturation of negative values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{common::TensorInfo, error::TensorError, ops::FloatUnaryOps, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-1.0]);
    let b = a.elu_(1.0, &mut a.clone())?;
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