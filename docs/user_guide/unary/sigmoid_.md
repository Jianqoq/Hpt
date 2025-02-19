# sigmoid_
```rust
sigmoid_(x: &Tensor<T>, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \frac{1}{1 + e^{-x}}$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError, TensorInfo};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.sigmoid_(&mut a.clone())?;
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