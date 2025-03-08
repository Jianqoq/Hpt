# par_iter_mut
```rust
fn par_iter_mut(x: &mut Tensor<T>) -> ParStridedMut<T>
```

similar as `par_iter`, input pass to the closure will be mutable. You can use it to do inplace computation.

## Parameters:

x: Tensor to iterate

## Returns:

`ParStridedMut`

## Examples:
```rust
use hpt::Tensor;
use hpt::iter::TensorIterator;
use hpt::iter::rayon::iter::ParallelIterator;

fn main() -> anyhow::Result<()> {
    let mut x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    x.par_iter_mut().for_each(|x|{
        *x = x.sin();
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