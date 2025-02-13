# triu
```rust
triu(
    x: &Tensor<T>,
    k: i64
) -> Result<Tensor<T>, TensorError>
```
Returns a copy of the tensor with elements below the k-th diagonal zeroed.

## Parameters:
`x`: Input tensor of at least 2-D

`k`: Diagonal below which to zero elements. k=0 is the main diagonal, k>0 is above and k<0 is below the main diagonal

## Returns:
A new tensor with the same shape as input, where all elements below the k-th diagonal are zero.

## Examples:
```rust
use hpt::{Tensor, TensorError, ShapeManipulate, TensorCreator};
fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 
                                4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0]).reshape(&[3, 3])?;
    // Main diagonal (k=0)
    let b = a.triu(0)?;
    println!("{}", b);
    // [[1, 2, 3],
    //  [0, 5, 6],
    //  [0, 0, 9]]

    // One diagonal above main (k=1)
    let c = a.triu(1)?;
    println!("{}", c);
    // [[0, 2, 3],
    //  [0, 0, 6],
    //  [0, 0, 0]]

    Ok(())
}
```