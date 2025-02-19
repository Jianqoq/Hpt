use safetensors::SafeTensors;

#[diagnostic::on_unimplemented(
    message = "Cannot perform operation on type `{Self}` because it doesn't implement required features"
)]
pub trait FromSafeTensors {
    fn from_safe_tensors(data: &SafeTensors, tensor_name: &str) -> Self;
}

impl<T: FromSafeTensors> FromSafeTensors for Option<T> {
    fn from_safe_tensors(data: &SafeTensors, tensor_name: &str) -> Self {
        Some(T::from_safe_tensors(data, tensor_name))
    }
}
