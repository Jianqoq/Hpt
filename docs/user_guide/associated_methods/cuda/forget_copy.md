# forget_copy
```rust
unsafe fn forget_copy(x: &Tensor<T, Cuda>) -> Result<(cudarc::driver::CudaSlice<u8>, std::alloc::Layout), TensorError>
```
clone the current Tensor data and return raw data.

## Note
Similar as `forget`, but `forget_copy` doesn't need to check reference count

## Parameters:
`x`: The input tensor

## Returns:
`cudarc::driver::CudaSlice<u8>`: A slice pointing to the cloned tensor's data
`std::alloc::Layout`: Can be used to check the byte size

## Examples:
```rust
use hpt::{backend::Cuda, error::TensorError, ops::TensorCreator, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32, Cuda>::empty([1, 2, 3])?;
    let ret = unsafe { a.forget_copy() }?;
    Ok(())
}
```