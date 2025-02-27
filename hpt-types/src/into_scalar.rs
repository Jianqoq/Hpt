use crate::convertion::Convertor;
use half::{bf16, f16};
use hpt_macros::impl_into_scalar;
use num_complex::{Complex32, Complex64};
/// A trait for converting a scalar into another scalar type.
pub trait Cast<T> {
    /// Convert the scalar into another scalar type.
    fn cast(self) -> T;
}

impl_into_scalar!();

#[cfg(feature = "cuda")]
mod cud_impl {
    use super::*;
    use crate::cuda_types::convertion::CudaConvertor;
    use crate::cuda_types::scalar::Scalar;
    use hpt_macros::impl_into_cuda_scalar;
    impl_into_cuda_scalar!();
}
