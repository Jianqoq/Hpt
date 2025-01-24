# tril
```rust
tril(
    x: &Tensor<T>,
    k: i64
) -> Result<Tensor<T>, TensorError>
```
Returns a copy of the tensor with elements above the k-th diagonal zeroed.

## Parameters:
`x`: Input tensor of at least 2-D
`k`: Diagonal above which to zero elements. k=0 is the main diagonal, k>0 is above and k<0 is below the main diagonal

## Returns:
A new tensor with the same shape as input, where all elements above the k-th diagonal are zero.

## Examples:
```rust
use tensor_dyn::{Tensor, TensorError, ShapeManipulate, TensorCreator};
fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 
                                4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0]).reshape(&[3, 3])?;
    // Main diagonal (k=0)
    let b = a.tril(0)?;
    println!("{}", b);
    // [[1, 0, 0],
    //  [4, 5, 0],
    //  [7, 8, 9]]

    // One diagonal below main (k=-1)
    let c = a.tril(-1)?;
    println!("{}", c);
    // [[0, 0, 0],
    //  [4, 0, 0],
    //  [7, 8, 0]]

    Ok(())
}
```