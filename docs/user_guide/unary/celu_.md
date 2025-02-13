# celu_
```rust
celu_(x: &Tensor<T>, alpha: C, out: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{max}(0, x) + \text{min}(0, \alpha \cdot (e^{x/\alpha} - 1))$ for all elements with out

## Parameters:
`x`: Input values
`alpha`: Parameter controlling the saturation of negative values
`out`: Tensor to write to

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-1.0]);
    let b = a.celu_(1.0, &a)?;
    println!("{}", b);  // prints: -0.6321206
    Ok(())
}
```