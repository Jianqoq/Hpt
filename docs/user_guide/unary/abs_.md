# abs_
```rust
abs_(
    x: &Tensor<T>, 
    out: &mut Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Calculate absolute of input with out
## Parameters:
`x`: input Tensor
`out`: Tensor to write to
## Returns:
Tensor with type `T`
## Examples:
```rust
use hpt::{NormalUaryOps, Tensor, TensorError, TensorInfo};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-10.0]);
    let b = a.abs_(&mut a.clone())?;
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