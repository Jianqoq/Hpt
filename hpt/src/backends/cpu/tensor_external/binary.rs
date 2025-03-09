use crate::backend::Cpu;
use crate::backends::cpu::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::binary::NormalBinOps, tensor::CommonBounds};
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
        A: CommonBounds + NormalOut<B>,
        B: CommonBounds,
        <A as NormalOut<B>>::Output: CommonBounds,
        <A as NormalOut<B>>::Output: Cast<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
        Al: Allocator,
        Al::Output: AllocatorOutputRetrive,
    {
        type Output = $output<NormalType<A, B>, Cpu, DEVICE, Al>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>, Cpu, DEVICE, Al>;

        #[track_caller]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._add(b), |a, b| a._add(b), Some(out))
        }
        #[track_caller]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._sub(b), |a, b| a._sub(b), Some(out))
        }
        #[track_caller]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._mul(b), |a, b| a._mul(b), Some(out))
        }
        #[track_caller]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: BorrowMut<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._rem(b), |a, b| a._rem(b), Some(out))
        }
    }
    };
}

impl_bin_ops!(
    [_Tensor<A, Cpu, DEVICE, Al>],
    [&_Tensor<B, Cpu, DEVICE, Al>],
    _Tensor
);
impl_bin_ops!(
    [_Tensor<A, Cpu, DEVICE, Al>],
    [_Tensor<B, Cpu, DEVICE, Al>],
    _Tensor
);
impl_bin_ops!(
    [&_Tensor<A, Cpu, DEVICE, Al>],
    [&_Tensor<B, Cpu, DEVICE, Al>],
    _Tensor
);
impl_bin_ops!(
    [&_Tensor<A, Cpu, DEVICE, Al>],
    [_Tensor<B, Cpu, DEVICE, Al>],
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
        A: CommonBounds + NormalOut<B>,
        B: CommonBounds,
        <A as NormalOut<B>>::Output: CommonBounds,
        <A as NormalOut<B>>::Output: Cast<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
        Al: Allocator,
        Al::Output: AllocatorOutputRetrive,
    {
        type Output = Tensor<NormalType<A, B>, Cpu, DEVICE, Al>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = Tensor<NormalType<A, B>, Cpu, DEVICE, Al>;
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
    [Tensor<A, Cpu, DEVICE, Al>],
    [&Tensor<B, Cpu, DEVICE, Al>],
    Tensor
);
impl_bin_ops_basic!(
    [Tensor<A, Cpu, DEVICE, Al>],
    [Tensor<B, Cpu, DEVICE, Al>],
    Tensor
);
impl_bin_ops_basic!(
    [&Tensor<A, Cpu, DEVICE, Al>],
    [&Tensor<B, Cpu, DEVICE, Al>],
    Tensor
);
impl_bin_ops_basic!(
    [&Tensor<A, Cpu, DEVICE, Al>],
    [Tensor<B, Cpu, DEVICE, Al>],
    Tensor
);
