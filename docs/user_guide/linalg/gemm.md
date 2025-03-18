# gemm
```rust
gemm(
    a: &Tensor<T>,
    b: &Tensor<T> | Tensor<T>,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Result<Tensor<T>, TensorError>
```
Perform general matrix multiplication of two tensors. The behavior depends on the dimensions of the input tensors:

- If both tensors are 2D, they are multiplied as matrices
- If either tensor is ND (N > 2), it is treated as a stack of matrices
- Broadcasting is applied to match dimensions

## Parameters:
`a`: First input tensor.

`b`: Second input tensor.

`alpha`: Scaling factor for the matrix product (A @ B)

`beta`: Scaling factor for the existing values in output matrix C

`conj_dst`: Whether to conjugate C before scaling with beta

`conj_lhs`: Whether to conjugate A before multiplication

`conj_rhs`: Whether to conjugate B before multiplication

## Returns:
A new Tensor containing the result of general matrix multiplication.

## Examples:
```rust
use hpt::{
    error::TensorError,
    ops::{Gemm, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // 2D matrix multiplication
    let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
    println!("2D result:\n{}", c);

    // 3D batch matrix multiplication
    let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    let f = d.gemm(&e, 0.0, 1.0, false, false, false)?; // 2 matrices of shape 2x2
    println!("3D result:\n{}", f);

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |