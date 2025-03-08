# par_iter_simd
```rust
fn par_iter_simd(x: &Tensor<T>) -> ParStridedSimd<T>
```

Convert Tensor to `ParStridedSimd` iterator which will use simd register to do computation when it is possible, `ParStridedSimd` will split the tasks and execute the method the user provides

## Parameters:

x: Tensor to iterate

## Returns:

`ParStridedSimd`

## Note

- Simd closure will only be execute when the result simd type number of lane is equal to input's.

- In Simd closure, result can only call `write_unaligned` method to write data.

## Examples:
```rust
use hpt::iter::TensorIterator;
use hpt::types::math::FloatOutUnary;
use hpt::Tensor;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    let res = x
        .par_iter_simd()
        .strided_map_simd(
            |(res, x)| {
                *res = x.sin();
            },
            |(res, x)| {
                res.write_unaligned(x._sin());
            },
        )
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