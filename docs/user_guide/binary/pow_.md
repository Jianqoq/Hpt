# pow_
```rust
pow_(
    x: Tensor<A>, 
    y: &Tensor<B> | Tensor<B> | scalar,
    out: &mut Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large x ^ y$ for all elements with out

## Parameters:
`x`: First input tensor

`y`: Second input tensor or scalar exponent

`out`: Tensor to write to

## Examples:
```rust
use hpt::{error::TensorError, ops::FloatBinOps, Tensor};

fn main() -> Result<(), TensorError> {
    // Create input tensors
    let a = Tensor::<f32>::new(&[2.0, 3.0, 4.0]);
    let b = Tensor::<f32>::new(&[2.0, 3.0, 2.0]);

    // Compute power and store result in a
    let c = a.pow_(&b, &mut a.clone())?;
    println!("{}", c);
    // Output:
    // [4.0, 27.0, 16.0]

    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅        |
| Cuda    | ✅        |