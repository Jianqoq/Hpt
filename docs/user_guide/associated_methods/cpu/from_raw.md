# from_raw
```rust
unsafe fn from_raw<S: Into<Shape>>(data: *mut T, shape: S) -> Result<Tensor<T>, TensorError>
```
Creates a new tensor from an existing raw pointer and shape without taking ownership of the memory.

## Note
This method allows you to create a tensor that uses an existing memory allocation. The tensor will use the provided memory without allocating new memory or taking ownership of it. This is useful for integrating with external libraries or when you want to manually manage memory.

## Parameters:
`data`: A `*mut T` pointing to the pre-allocated memory on the CPU device

`shape`: The shape to use for the tensor, which can be any type that can be converted to Shape

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{common::Shape, error::TensorError, Tensor};

fn main() -> Result<(), TensorError> {
    let origin_layout = std::alloc::Layout::from_size_align(6 * std::mem::size_of::<f32>(), 64).unwrap();
    let data = unsafe { std::alloc::alloc(origin_layout) };

    // Create a 2x3 tensor using the existing memory
    let shape = Shape::new([2, 3]);
    let tensor = unsafe { Tensor::<f32>::from_raw(data as *mut f32, shape)? };

    println!("{}", tensor); // Should print the 2x3 tensor with the values

    // Note: When the tensor is dropped, it will NOT deallocate the memory
    // since it did not take ownership of it

    // it is better to use forget method to transfer the ownership of the memory to the caller
    // the forget method will track the reference count of the memory
    let (raw_ptr, layout) = unsafe { tensor.forget() }?;
    assert_eq!(origin_layout, layout);
    unsafe { std::alloc::dealloc(raw_ptr, layout) };
    Ok(())
}
```