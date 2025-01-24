# permute_inv
```rust
permute_inv(
    x: &Tensor<T>,
    axes: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
) -> Result<Tensor<T>, TensorError>
```
Performs the inverse permutation of dimensions according to the given axes order. This is equivalent to undoing a previous permutation.

## Parameters:
`x`: Input tensor

`axes`: The permutation to invert. Must be a permutation of [0, 1, ..., n-1] where n is the number of dimensions.

## Returns:
A new tensor with the dimensions reordered according to the inverse of the specified permutation.

## Examples:
```rust
use tensor_dyn::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [2, 3, 4]
    let a = Tensor::<f32>::zeros(&[2, 3, 4])?;

    // Permute dimensions to [4, 2, 3]
    let b = a.permute(&[2, 0, 1])?;
    println!("{}", b.shape()); // prints [4, 2, 3]

    // Apply inverse permutation to get back original shape
    let c = b.permute_inv(&[2, 0, 1])?;
    println!("{}", c.shape()); // prints [2, 3, 4]

    // Verify that permute_inv undoes permute
    assert_eq!(c.shape(), a.shape());

    Ok(())
}
```