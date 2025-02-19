# randn
```rust
randn_like(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
same as `randn` but the shape will be based on `x`.
## Parameters:
`x`: input Tensor
## Returns:
Tensor with type `T`
## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::randn([10, 10])?;
    println!("{}", a.randn_like()?);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |