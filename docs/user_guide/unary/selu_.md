# selu_
```rust
selu_(
    x: &Tensor<T>, 
    alpha: C | None, 
    gamma: C | None,
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Compute $\large \lambda * (\alpha * (e^x - 1))$ for $x < 0$, $\large \lambda * x$ for $x \geq 0$ for all elements

## Parameters:
`x`: Input values

`alpha`: Scale for the negative part, default is `1.6732632423543772848170429916717`

`gamma`: Scale for both positive and negative parts, default is `1.0507009873554804934193349852946`

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.selu_(None, None, &a)?;
    println!("{}", b);
    Ok(())
}
```