# from_raw
```rust
unsafe fn from_raw<S: Into<Shape>>(data: CudaSlice<T>, shape: S) -> Result<Tensor<T>, TensorError>
```
Creates a new tensor from an existing raw pointer and shape without taking ownership of the memory.

## Note
This method allows you to create a tensor that uses an existing memory allocation. The tensor will use the provided memory without allocating new memory or taking ownership of it. This is useful for integrating with external libraries or when you want to manually manage memory.

## Parameters:
`data`: A `CudaSlice<T>` pointing to the pre-allocated memory on the CUDA device

`shape`: The shape to use for the tensor, which can be any type that can be converted to Shape

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::backend::Cuda;
use hpt::re_exports::cudarc;
use hpt::{common::Shape, error::TensorError, Tensor};

fn main() -> Result<(), TensorError> {
    // In a real scenario, this would come from an external source
    // or be created through another API
    let device = cudarc::driver::CudaDevice::new(0)?;
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cuda_slice = device.htod_copy(values)?;

    // Create a 2x3 tensor using the existing memory
    let shape = Shape::new([2, 3]);
    let tensor = unsafe { Tensor::<f32, Cuda>::from_raw(cuda_slice, shape)? };

    println!("{}", tensor); // Should print the 2x3 tensor with the values

    // Note: When the tensor is dropped, it will NOT deallocate the memory
    // since it did not take ownership of it

    // you will need to call `forget`
    // the forget method will track the reference count of the memory
    let (raw_ptr, layout) = unsafe { tensor.forget() }?;
    // raw_ptr drop will be called when tensor goes out of scope, memory will be deallocated
    Ok(())
}
```