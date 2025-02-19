# par_iter_mut_simd
```rust
fn par_iter_mut_simd(x: &mut Tensor<T>) -> ParStridedMutSimd<T>
```

similar as `par_iter_simd`, input pass to the closure will be mutable. You can use it to do inplace computation.

## Parameters:

x: Tensor to iterate

## Returns:

`ParStridedMutSimd`

## Note

- Simd closure will only be execute when the result simd type number of lane is equal to input's.

- In Simd closure, result can only call `write_unaligned` method to write data.

## Examples:
```rust
use hpt::*;

fn main() -> anyhow::Result<()> {
    let mut x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    x.par_iter_mut_simd().for_each(|x|{
        *x = x.sin();
    },|x|{
        let val = x.read_unaligned();
        x.write_unaligned(val._sin());
    });

    println!("{}", x);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |