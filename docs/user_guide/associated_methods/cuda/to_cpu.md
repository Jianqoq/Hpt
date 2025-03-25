# to_cpu
```rust
fn to_cpu<const DEVICE_ID: usize>(x: &Tensor<T, Cuda>) -> Result<Tensor<T, Cpu>, TensorError>
```
Transfers a tensor from CUDA GPU memory to CPU memory, creating a new tensor in host memory.

currently only `DEVICE_ID` = 0 is supported

## Parameters:
`DEVICE_ID`: A compile-time constant specifying the target CPU device ID (default is 0)

## Returns:
A new `Tensor<T>` located on the specified CPU device, or a TensorError if the transfer fails.

## Examples:
```rust
use hpt::{backend::Cuda, error::TensorError, ops::TensorCreator, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32, Cuda>::empty([1, 2, 3])?;
    println!("{}", a.to_cpu::<0>()?);
    Ok(())
}
```