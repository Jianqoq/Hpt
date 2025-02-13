# gather
```rust
gather(
    x: &Tensor<T>,
    indices: &Tensor<I>,
    axis: i64
) -> Result<Tensor<T>, TensorError>
```
Gathers values along an axis specified by `axis` using the indices specified by `indices`.

The `indices` tensor selects the elements to be gathered from `x` along the specified axis. The output tensor has the same rank as `x` with shape identical to `x` except for the dimension specified by `axis` which has the same size as `indices`.

## Parameters:
`x`: Input tensor from which to gather values.

`indices`: Index tensor that specifies the indices of elements to gather.

`axis`: The axis along which to gather values. Supports negative indexing.

## Returns:
A new tensor containing the gathered values.

## Examples:
```rust
use hpt::{Tensor, TensorCreator, TensorError, AdvancedOps};

fn main() -> Result<(), TensorError> {
    // Create a sample tensor
    let x = Tensor::new(&[[1., 2., 3.], 
                         [4., 5., 6.]]);
    
    // Gather elements using indices
    let indices = Tensor::new(&[0, 2]);
    let result = x.gather(&indices, 1)?;  // gather along columns
    println!("Gathered values:\n{}", result);
    // Output will be: [[1., 3.], [4., 6.]]
    
    Ok(())
}
```