# forget_copy
```rust
unsafe fn forget_copy(x: &Tensor<T, Cpu>) -> Result<(*mut u8, std::alloc::Layout), TensorError>
```
clone the current Tensor data and return raw data.

## Note
Similar as `forget`, but `forget_copy` doesn't need to check reference count

## Parameters:
`x`: The input tensor

## Returns:
`*mut u8`: A pointer pointing to the cloned tensor's data
`std::alloc::Layout`: Can be used to check the byte size

## Examples:
```rust
use hpt::{error::TensorError, ops::TensorCreator, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::empty([1, 2, 3])?;
    let (ptr, layout) = unsafe { a.forget_copy() }?;
    unsafe { std::alloc::dealloc(ptr, layout) };
    Ok(())
}
```