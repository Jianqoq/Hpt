# matmul
```rust
matmul(
    a: &Tensor<T>,
    b: &Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Perform matrix multiplication of two tensors. The behavior depends on the dimensions of the input tensors:

- If both tensors are 2D, they are multiplied as matrices
- If either tensor is ND (N > 2), it is treated as a stack of matrices
- Broadcasting is applied to match dimensions

Matmul supports more data type than `gemm` method.

## Parameters:
`a`: First input tensor.

`b`: Second input tensor.

## Returns:
A new Tensor containing the result of matrix multiplication.

## Examples:
```rust
use hpt::{
    error::TensorError,
    ops::{Matmul, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // 2D matrix multiplication
    let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    let c = a.matmul(&b)?;
    println!("2D result:\n{}", c);

    // 3D batch matrix multiplication
    let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    let f = d.matmul(&e)?; // 2 matrices of shape 2x2
    println!("3D result:\n{}", f);

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |