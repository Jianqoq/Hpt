# strided_map_simd
```rust
fn strided_map_simd(self, f: F, vec_f: F2) -> ParStridedMap
```

Applies two functions to process data: one for scalar operations and one for vectorized (SIMD) operations, enabling faster processing through hardware acceleration.

## Notes

- Automatically falls back to scalar function when SIMD is not possible

## Parameters

- `self`: The parallel SIMD iterator
  - Type: `ParStridedSimd<T>`
- `f`: The scalar mapping function
  - Type: `FnMut((&mut T, T))`
  - Used for non-vectorized elements
- `vec_f`: The SIMD mapping function
  - Type: `FnMut((&mut Simd<T>, Simd<T>))`
  - Used for vectorized operations

## Returns

A `ParStridedMapSimd` iterator

## Examples:
```rust
use hpt::*;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    let res = x.par_iter_simd().strided_map_simd(
        |(res, x)| {
            *res = x.sin();
        },
        |(res, x)| {
            res.write_unaligned(x._sin());
        },
    ).collect::<Tensor<f64>>();

    println!("{}", res);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |