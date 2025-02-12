# topk
```rust
topk(
    x: &Tensor<T>,
    k: i64,
    dim: i64,
    largest: bool,
    sorted: bool
) -> Result<(Tensor<I>, Tensor<T>), TensorError>
```
Returns the k largest or smallest elements along a specified dimension, and their indices.

## Parameters:
`x`: Input tensor.

`k`: Number of top elements to return.

`dim`: The dimension to sort along. Supports negative indexing.

`largest`: If true, returns the k largest elements; if false, the k smallest.

`sorted`: If true, the returned elements are sorted in descending/ascending order.

## Returns:
A tuple of two tensors:
- First tensor contains the indices of the top-k elements
- Second tensor contains the values of the top-k elements

## Examples:
```rust
use hpt_core::{Tensor, TensorError, AdvancedOps};

fn main() -> Result<(), TensorError> {
    let x = Tensor::<f64>::new(&[5., 2., 8., 1., 9., 3.]);
    
    // Get top 3 largest values and their indices
    let (indices, values) = x.topk(3, 0, true, true)?;
    println!("Top 3 values: {}", values);    // [9., 8., 5.]
    println!("Their indices: {}", indices);   // [4, 2, 0]
    
    // Get top 2 smallest values, unsorted
    let (indices, values) = x.topk(2, 0, false, false)?;
    println!("Bottom 2 values: {}", values);  // Values might be in any order
    
    Ok(())
}
```