use hpt::{error::TensorError, ops::TensorCreator, Tensor};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::empty([1, 2, 3])?;
    let (ptr, layout) = unsafe { a.forget_copy() }?;
    unsafe { std::alloc::dealloc(ptr, layout) };
    Ok(())
}
