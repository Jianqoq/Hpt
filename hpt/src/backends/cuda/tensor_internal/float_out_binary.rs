use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_traits::{ops::binary::FloatBinOps, tensor::CommonBounds};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType, type_promote::FloatOutBinary};

use crate::{
    backend::Cuda, backends::cuda::utils::binary::binary_normal::binary_fn_with_out_simd,
    tensor_base::_Tensor,
};

type FloatBinaryType<T, B> = <T as FloatOutBinary<B>>::Output;

impl<T, B, const DEVICE: usize, Al> FloatBinOps<_Tensor<B, Cuda, DEVICE, Al>>
    for _Tensor<T, Cuda, DEVICE, Al>
where
    B: CommonBounds + DeviceRepr + CudaType,
    T: FloatOutBinary<B> + CommonBounds + DeviceRepr + CudaType,
    FloatBinaryType<T, B>: CommonBounds + DeviceRepr + CudaType,
    Scalar<T>: FloatOutBinary<Scalar<B>, Output = Scalar<FloatBinaryType<T, B>>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatBinaryType<T, B>, Cuda, DEVICE, Al>;

    type OutputMeta = FloatBinaryType<T, B>;

    type InplaceOutput = _Tensor<FloatBinaryType<T, B>, Cuda, DEVICE, Al>;

    fn hypot<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cuda, DEVICE, Al>>,
    {
        binary_fn_with_out_simd(
            "hypot",
            self,
            &rhs.into(),
            |out, a, b| out.assign(a._hypot(b)),
            None::<Self::InplaceOutput>,
        )
    }

    fn hypot_<C, U>(
        &self,
        rhs: C,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cuda, DEVICE, Al>>,
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        binary_fn_with_out_simd(
            "hypot",
            self,
            &rhs.into(),
            |out, a, b| out.assign(a._hypot(b)),
            Some(out),
        )
    }

    fn div_<C, U>(
        &self,
        rhs: C,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cuda, DEVICE, Al>>,
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        binary_fn_with_out_simd(
            "div",
            self,
            &rhs.into(),
            |out, a, b| out.assign(a._div(b)),
            Some(out),
        )
    }

    fn pow<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cuda, DEVICE, Al>>,
    {
        binary_fn_with_out_simd(
            "pow",
            self,
            &rhs.into(),
            |out, a, b| out.assign(a._pow(b)),
            None::<Self::InplaceOutput>,
        )
    }

    fn pow_<C, U>(
        &self,
        rhs: C,
        out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<_Tensor<B, Cuda, DEVICE, Al>>,
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        binary_fn_with_out_simd(
            "pow",
            self,
            &rhs.into(),
            |out, a, b| out.assign(a._pow(b)),
            Some(out),
        )
    }
}
