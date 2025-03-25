# to_cuda
```rust
fn to_cuda<const DEVICE_ID: usize>(x: &Tensor<T, Cpu>) -> Result<Tensor<T, Cuda>, TensorError>
```
Transfers a tensor from CPU memory to CUDA GPU memory, creating a new tensor on the specified CUDA device.

## Parameters:
`DEVICE_ID`: A compile-time constant specifying the target CUDA device ID (default is 0)

## Returns:
A new `Tensor<T>` located on the specified CUDA device, or a TensorError if the transfer fails.

## Examples:
```rust
use hpt::{error::TensorError, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([1.5, 2.7, 3.2]).to_cuda::<0>()?;
    println!("{}", a);
    Ok(())
}
```