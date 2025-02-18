use cudarc::driver::DeviceRepr;
use hpt_traits::{CommonBounds, FloatBinOps};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType, type_promote::FloatOutBinary};

use crate::{
    ops::{
        cpu::tensor_internal::float_out_unary::FloatBinaryType,
        cuda::utils::binary::binary_normal::binary_fn_with_out_simd,
    },
    tensor_base::_Tensor,
    Cuda,
};

impl<T, const DEVICE: usize> FloatBinOps for _Tensor<T, Cuda, DEVICE>
where
    T: CommonBounds + DeviceRepr + CudaType + FloatOutBinary,
    Scalar<T>: FloatOutBinary<Output = Scalar<FloatBinaryType<T>>>,
    FloatBinaryType<T>: CommonBounds + DeviceRepr + CudaType,
{
    type Output = _Tensor<FloatBinaryType<T>, Cuda, DEVICE>;

    type OutputMeta = FloatBinaryType<T>;

    type InplaceOutput = _Tensor<FloatBinaryType<T>, Cuda, DEVICE>;

    fn hypot(
        &self,
        rhs: &Self,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError> {
        binary_fn_with_out_simd(
            "hypot",
            self,
            rhs,
            |out, a, b| out.assign(a._hypot(b)),
            None::<Self::InplaceOutput>,
        )
    }

    fn hypot_<U>(
        &self,
        rhs: &Self,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        binary_fn_with_out_simd(
            "hypot",
            self,
            rhs,
            |out, a, b| out.assign(a._hypot(b)),
            Some(out),
        )
    }

    fn div_<U>(
        &self,
        rhs: &Self,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        binary_fn_with_out_simd(
            "div",
            self,
            rhs,
            |out, a, b| out.assign(a._div(b)),
            Some(out),
        )
    }
}
