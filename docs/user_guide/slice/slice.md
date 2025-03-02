# Slice

```rust
slice(x: &Tensor<T>, index: &[(i64, i64, i64)]) -> Result<Tensor<T>, TensorError>
```

Create a new Tensor by slicing an existing Tensor. Slicing allows you to extract a portion of a tensor using index ranges for each dimension.

## Parameters:
`x`: Input tensor to be sliced.

`index`: `(start, end, step)`: Select from start to end with step

## Returns:
A new Tensor containing the sliced values.

## Examples:
```rust
use hpt::{ShapeManipulate, Slice, Tensor, TensorCreator, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::arange(0, 16)?.reshape(&[4, 4])?;

    // Select rows 1:3, all columns
    let b = a.slice(&[(1, 3, 1), (0, 4, 1)])?;

    // Select all rows with step 2, columns 0:2
    let c = a.slice(&[(2, 4, 1), (0, 2, 1)])?;

    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
    Ok(())
}
```

## Note

To simplify the slicing, hpt provides a macro that allows you to write python like slicing syntax.

## Examples:
```rust
use hpt::{select, ShapeManipulate, Slice, Tensor, TensorCreator, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::arange(0, 4 * 4 * 4 * 4)?.reshape(&[4, 4, 4, 4])?;

    // Select rows 1:3, full range except the last dim, last dim is 3:4
    let b = a.slice(&select![1:3, .., 3:4])?;

    println!("{}", b);
    Ok(())
}
```