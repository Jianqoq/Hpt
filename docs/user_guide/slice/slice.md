# Slice

```rust
slice(x: &Tensor<T>, index: &[Slice]) -> Result<Tensor<T>, TensorError>
```

Create a new Tensor by slicing an existing Tensor. Slicing allows you to extract a portion of a tensor using index ranges for each dimension.

## Parameters:
`x`: Input tensor to be sliced.

`index`: Array of Slice objects specifying how to slice each dimension:
- `From(i)`: Select start from index i

- `Full`: Select entire dimension (equivalent to `[:]`)

- `RangeFrom(i)`: Select from index i to end (equivalent to `[i:]`)

- `RangeTo(i)`: Select from start to index i (equivalent to `[:i]`)

- `Range((start, end))`: Select from start to end (equivalent to `[start:end]`)

- `StepByRangeFrom((start, step))`: Select from start with step (equivalent to `[start::step]`)

- `StepByFullRange(step)`: Select entire dimension with step (equivalent to `[::step]`)

- `StepByRangeFromTo((start, end, step))`: Select from start to end with step (equivalent to `[start:end:step]`)

- `StepByRangeTo((end, step))`: Select from start to end with step (equivalent to `[:end:step]`)

## Returns:
A new Tensor containing the sliced values.

## Examples:
```rust
use hpt::{ShapeManipulate, Slice, Tensor, TensorCreator, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::arange(0, 16)?.reshape(&[4, 4])?;

    // Select rows 1:3, all columns
    let b = a.slice(&[Slice::Range((1, 3)), Slice::Full])?;

    // Select all rows with step 2, columns 0:2
    let c = a.slice(&[Slice::StepByFullRange(2), Slice::RangeTo(2)])?;

    println!("{}", b);
    println!("{}", c);
    Ok(())
}
```

## Note

To simplify the slicing, hpt provides a macro that allows you to write python like slicing syntax.

## Examples:
```rust
use hpt::{match_selection, ShapeManipulate, Slice, Tensor, TensorCreator, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::arange(0, 16)?.reshape(&[4, 4])?;

    // Select rows 1:3, all columns
    let b = a.slice(&match_selection![1:3, :])?;

    // Select all rows with step 2, columns 0:2
    let c = a.slice(&match_selection![::2, :2])?;

    println!("{}", b);
    println!("{}", c);
    Ok(())
}
```