# strided_map
```rust
fn strided_map(self, f: F) -> ParStridedMap
```

Applies a function to each element of the iterator with strided access pattern. Useful for parallel data transformation operations.

## Parameters

- `self`: The parallel iterator
  - Type: `ParStrided<T>` or `ParStridedZip<T>`
- `f`: The mapping function
  - Type: `FnMut((&mut T, &T))`
  - Requirements: Must be thread-safe (`Send + Sync`)

## Returns

A `ParStridedMap` iterator

## Examples:
```rust
use hpt::Tensor;
use hpt::iter::TensorIterator;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    let res = x.par_iter().strided_map(|(res, x)|{
        *res = x.sin();
    }).collect::<Tensor<f64>>();

    println!("{}", res);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |