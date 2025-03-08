# collect
```rust
fn collect<U>(strided_map) -> U
```

Convert map struct into `U` type, `U` must be `Tensor` type.

## Parameters:

strided_map: Map struct, in hpt, there are couple of map struct

## Returns:

Tensor

## Examples:
```rust
use hpt::iter::TensorIterator;
use hpt::Tensor;
fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    let res = x
        .par_iter()
        .strided_map(|(res, x)| {
            *res = x.sin();
        })
        .collect::<Tensor<f64>>();

    println!("{}", res);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |