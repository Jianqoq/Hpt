use hpt_traits::{CommonBounds, FloatBinOps};
use hpt_types::{dtype::TypeCommon, type_promote::FloatOutBinary};

use crate::{
    ops::cpu::utils::binary::binary_normal::binary_fn_with_out_simd, tensor_base::_Tensor, Cpu,
};

use super::float_out_unary::FloatBinaryType;

impl<T, const DEVICE: usize> FloatBinOps for _Tensor<T, Cpu, DEVICE>
where
    T: FloatOutBinary + CommonBounds,
    FloatBinaryType<T>: CommonBounds,
    T::Vec: FloatOutBinary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>,
{
    type Output = _Tensor<FloatBinaryType<T>, Cpu, DEVICE>;

    type OutputMeta = FloatBinaryType<T>;

    type InplaceOutput = _Tensor<FloatBinaryType<T>, Cpu, DEVICE>;

    fn hypot(
        &self,
        rhs: &Self,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError> {
        binary_fn_with_out_simd(
            self,
            rhs,
            |a, b| a._hypot(b),
            |a, b| a._hypot(b),
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
        binary_fn_with_out_simd(self, rhs, |a, b| a._hypot(b), |a, b| a._hypot(b), Some(out))
    }

    fn div_<U>(
        &self,
        rhs: &Self,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        binary_fn_with_out_simd(self, rhs, |a, b| a._div(b), |a, b| a._div(b), Some(out))
    }
}
