use crate::backend::Cuda;
use crate::backends::cuda::utils::binary::binary_normal::binary_fn_precompiled;
use crate::re_exports::cudarc::driver::DeviceRepr;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_cudakernels::*;
use hpt_traits::{ops::binary::NormalBinOps, tensor::CommonBounds};
use hpt_types::dtype::CudaType;
use hpt_types::dtype::TypeCommon;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};
use std::borrow::BorrowMut;

/// a type alias for the output type of the binary operations of `A` and `B`
pub(crate) type NormalType<A, B> = <A as NormalOut<B>>::Output;

macro_rules! impl_bin_ops {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
    impl<A, B, const DEVICE: usize, Al> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where
        A: CommonBounds + NormalOut<B> + DeviceRepr + CudaType,
        B: CommonBounds + DeviceRepr + CudaType,
        <A as NormalOut<B>>::Output: CommonBounds + DeviceRepr + CudaType,
        <A as NormalOut<B>>::Output: Cast<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
        Al: Allocator,
        Al::Output: AllocatorOutputRetrive,
    {
        type Output = $output<NormalType<A, B>, Cuda, DEVICE, Al>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>, Cuda, DEVICE, Al>;

        #[track_caller]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_precompiled(
                &self,
                &rhs,
                "add",
                &ADD,
                Some(out)
            )
        }
        #[track_caller]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_precompiled(
                &self,
                &rhs,
                "sub",
                &SUB,
                Some(out)
            )
        }
        #[track_caller]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_precompiled(
                &self,
                &rhs,
                "mul",
                &MUL,
                Some(out)
            )
        }
        #[track_caller]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_precompiled(
                &self,
                &rhs,
                "rem",
                &REM,
                Some(out)
            )
        }
    }
    };
}

impl_bin_ops!(
    [_Tensor<A, Cuda, DEVICE, Al>],
    [&_Tensor<B, Cuda, DEVICE, Al>],
    _Tensor
);
impl_bin_ops!(
    [_Tensor<A, Cuda, DEVICE, Al>],
    [_Tensor<B, Cuda, DEVICE, Al>],
    _Tensor
);
impl_bin_ops!(
    [&_Tensor<A, Cuda, DEVICE, Al>],
    [&_Tensor<B, Cuda, DEVICE, Al>],
    _Tensor
);
impl_bin_ops!(
    [&_Tensor<A, Cuda, DEVICE, Al>],
    [_Tensor<B, Cuda, DEVICE, Al>],
    _Tensor
);

macro_rules! impl_bin_ops_basic {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
        impl<A, B, const DEVICE: usize, Al> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where
        A: CommonBounds + NormalOut<B> + DeviceRepr + CudaType,
        B: CommonBounds + DeviceRepr + CudaType,
        <A as NormalOut<B>>::Output: CommonBounds + DeviceRepr + CudaType,
        <A as NormalOut<B>>::Output: Cast<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
        Al: Allocator,
        Al::Output: AllocatorOutputRetrive,
    {
        type Output = Tensor<NormalType<A, B>, Cuda, DEVICE, Al>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = Tensor<NormalType<A, B>, Cuda, DEVICE, Al>;
        #[track_caller]
        #[inline]
        fn add_<U>(&self, rhs: $($rhs)*, mut out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            Ok(self.inner.add_(rhs.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
        }
        #[track_caller]
        #[inline]
        fn sub_<U>(&self, rhs: $($rhs)*, mut out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            Ok(self.inner.sub_(rhs.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
        }
        #[track_caller]
        #[inline]
        fn mul_<U>(&self, rhs: $($rhs)*, mut out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            Ok(self.inner.mul_(rhs.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
        }
        #[track_caller]
        #[inline]
        fn rem_<U>(&self, rhs: $($rhs)*, mut out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            Ok(self.inner.rem_(rhs.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
        }
    }
    };
}

impl_bin_ops_basic!(
    [Tensor<A, Cuda, DEVICE, Al>],
    [&Tensor<B, Cuda, DEVICE, Al>],
    Tensor
);
impl_bin_ops_basic!(
    [Tensor<A, Cuda, DEVICE, Al>],
    [Tensor<B, Cuda, DEVICE, Al>],
    Tensor
);
impl_bin_ops_basic!(
    [&Tensor<A, Cuda, DEVICE, Al>],
    [&Tensor<B, Cuda, DEVICE, Al>],
    Tensor
);
impl_bin_ops_basic!(
    [&Tensor<A, Cuda, DEVICE, Al>],
    [Tensor<B, Cuda, DEVICE, Al>],
    Tensor
);
