use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_traits::{ops::binary::FloatBinOps, tensor::CommonBounds};
use hpt_types::{dtype::TypeCommon, type_promote::FloatOutBinary};

use crate::{
    backends::cpu::utils::binary::binary_normal::binary_fn_with_out_simd, tensor_base::_Tensor,
};

type FloatBinaryType<T, B> = <T as FloatOutBinary<B>>::Output;

impl<T, B, A2, const DEVICE: usize> FloatBinOps<_Tensor<B, Cpu, DEVICE, A2>>
    for _Tensor<T, Cpu, DEVICE, A2>
where
    B: CommonBounds,
    T: FloatOutBinary<B> + CommonBounds,
    FloatBinaryType<T, B>: CommonBounds,
    T::Vec: FloatOutBinary<B::Vec, Output = <FloatBinaryType<T, B> as TypeCommon>::Vec>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatBinaryType<T, B>, Cpu, DEVICE, A2>;

    type OutputMeta = FloatBinaryType<T, B>;

    type InplaceOutput = _Tensor<FloatBinaryType<T, B>, Cpu, DEVICE, A2>;

    fn hypot<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cpu, DEVICE, A2>>,
    {
        binary_fn_with_out_simd(
            self,
            &rhs.into(),
            |a, b| a._hypot(b),
            |a, b| a._hypot(b),
            None::<Self::InplaceOutput>,
        )
    }

    fn hypot_<C, U>(
        &self,
        rhs: C,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
        C: Into<_Tensor<B, Cpu, DEVICE, A2>>,
    {
        binary_fn_with_out_simd(
            self,
            &rhs.into(),
            |a, b| a._hypot(b),
            |a, b| a._hypot(b),
            Some(out),
        )
    }

    fn div_<C, U>(
        &self,
        rhs: C,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
        C: Into<_Tensor<B, Cpu, DEVICE, A2>>,
    {
        binary_fn_with_out_simd(
            self,
            &rhs.into(),
            |a, b| a._div(b),
            |a, b| a._div(b),
            Some(out),
        )
    }

    fn pow<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cpu, DEVICE, A2>>,
    {
        binary_fn_with_out_simd(
            self,
            &rhs.into(),
            |a, b| a._pow(b),
            |a, b| a._pow(b),
            None::<Self::InplaceOutput>,
        )
    }

    fn pow_<C, U>(
        &self,
        rhs: C,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
        C: Into<_Tensor<B, Cpu, DEVICE, A2>>,
    {
        binary_fn_with_out_simd(
            self,
            &rhs.into(),
            |a, b| a._pow(b),
            |a, b| a._pow(b),
            Some(out),
        )
    }
}
