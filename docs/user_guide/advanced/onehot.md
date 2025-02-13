# onehot
```rust
onehot(
    x: &Tensor<T>,
    depth: usize,
    axis: i64,
    true_val: T,
    false_val: T
) -> Result<Tensor<T>, TensorError>
```
Creates a one-hot tensor from the input tensor. The output tensor will have an additional dimension of size `depth` inserted at `axis`, where indices from the input tensor select which index in this dimension gets the `true_val` value while all other indices get the `false_val` value.

## Parameters:
`x`: Input tensor containing indices.

`depth`: Size of the new dimension.

`axis`: Position to insert the new dimension. Supports negative indexing.

`true_val`: Value to place at the index specified by the input tensor.

`false_val`: Value to place at all other indices.

## Returns:
A new tensor with an additional dimension of size `depth`.

## Examples:
```rust
use hpt::{Tensor, TensorError, AdvancedOps};

fn main() -> Result<(), TensorError> {
    let indices = Tensor::<i64>::new(&[1, 0, 2]);
    
    // Create one-hot encoding with depth 3
    let onehot = indices.onehot(3, -1, 1, 0)?;
    println!("One-hot encoding:\n{}", onehot);
    // Output:
    // [[0, 1, 0],
    //  [1, 0, 0],
    //  [0, 0, 1]]
    
    Ok(())
}
```