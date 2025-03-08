# par_iter
```rust
fn par_iter(x: &Tensor<T>) -> ParStrided<T>
```

Convert Tensor to `ParStrided` iterator, `ParStrided` will split the tasks and execute the method the user provides

## Parameters:

x: Tensor to iterate

## Returns:

`ParStrided`

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