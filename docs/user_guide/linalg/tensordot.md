# tensordot
```rust
tensordot<const N: usize>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    axes: ([i64; N], [i64; N])
) -> Result<Tensor<T>, TensorError>
```
Compute tensor dot product along specified axes. This is a generalization of matrix multiplication to higher dimensions.

## Parameters:
`a`: First input tensor.

`b`: Second input tensor.

`axes`: A tuple of two arrays specifying the axes to contract over:
- First array contains axes from the first tensor
- Second array contains axes from the second tensor
- Arrays must have same length N

## Returns:
A new Tensor containing the result of the tensor dot product.

## Examples:
```rust
use hpt::{Tensor, TensorCreator, TensorDot, TensorError};

fn main() -> Result<(), TensorError> {
    // Matrix multiplication (2D tensordot)
    let a = Tensor::new(&[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&[[5., 6.], [7., 8.]]);
    let c = a.tensordot(&b, ([1], [0]))?;  // Contract last axis of a with first axis of b
    println!("Matrix multiplication:\n{}", c);

    // Higher dimensional example
    let d = Tensor::<f32>::ones(&[2, 3, 4])?;
    let e = Tensor::<f32>::ones(&[4, 3, 2])?;
    let f = d.tensordot(&e, ([1, 2], [1, 0]))?;  // Contract axes 1,2 of d with axes 1,0 of e
    println!("Higher dimensional result:\n{}", f);
    
    Ok(())
}
```