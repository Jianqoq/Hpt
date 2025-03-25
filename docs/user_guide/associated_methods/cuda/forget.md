# forget
```rust
unsafe fn forget(x: Tensor<T, Cuda>) -> Result<(cudarc::driver::CudaSlice<u8>, std::alloc::Layout), TensorError>
```
Transfers ownership of the tensor's memory to the caller, returning the raw pointer and layout information.

## Note
This method "forgets" about the tensor's memory without deallocating it, allowing you to take manual control of the memory management. Once called, the tensor's destructor will not free the underlying memory - the caller becomes responsible for proper deallocation.

## Parameters:
`x`: The input tensor whose memory ownership will be transferred

## Returns:
`cudarc::driver::CudaSlice<u8>`: A slice pointing to the tensor's data
`std::alloc::Layout`: Can be used to check the byte size

## Examples:
```rust
use hpt::{error::TensorError, Tensor};

fn main() -> Result<(), TensorError> {
    // Create a tensor with f32 values
    let a = Tensor::<f32>::new([1.5, 2.7, 3.2]).to_cuda::<0>()?;

    // Transfer ownership of the memory to the caller
    let (raw_ptr, layout) = unsafe { a.forget() }?;

    // raw_ptr drop will be called when a goes out of scope, memory will be deallocated
    Ok(())
}
```