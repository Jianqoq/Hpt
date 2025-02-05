use crate::ops::cpu::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::Cpu;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::binary::NormalBinOps, tensor::CommonBounds};
use hpt_types::dtype::TypeCommon;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};
use std::borrow::Borrow;

/// a type alias for the output type of the binary operations of `A` and `B`
pub(crate) type NormalType<A, B> = <A as NormalOut<B>>::Output;

macro_rules! impl_bin_ops {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
    impl<A, B, const DEVICE: usize> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where
        A: CommonBounds + NormalOut<B>,
        B: CommonBounds,
        <A as NormalOut<B>>::Output: CommonBounds,
        <A as NormalOut<B>>::Output: Cast<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
    {
        type Output = $output<NormalType<A, B>, Cpu, DEVICE>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>, Cpu, DEVICE>;

        #[cfg_attr(feature = "track_caller", track_caller)]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._add(b), |a, b| a._add(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._sub(b), |a, b| a._sub(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._mul(b), |a, b| a._mul(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._rem(b), |a, b| a._rem(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn pow(&self, rhs: $($rhs)*) -> std::result::Result<Self::Output, TensorError>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._pow(b), |a, b| a._pow(b), None::<Self::Output>)
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn pow_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._pow(b), |a, b| a._pow(b), Some(out))
        }
    }
    };
}

impl_bin_ops!(
    [_Tensor<A, Cpu, DEVICE>],
    [&_Tensor<B, Cpu, DEVICE>],
    _Tensor
);
impl_bin_ops!(
    [_Tensor<A, Cpu, DEVICE>],
    [_Tensor<B, Cpu, DEVICE>],
    _Tensor
);
impl_bin_ops!(
    [&_Tensor<A, Cpu, DEVICE>],
    [&_Tensor<B, Cpu, DEVICE>],
    _Tensor
);
impl_bin_ops!(
    [&_Tensor<A, Cpu, DEVICE>],
    [_Tensor<B, Cpu, DEVICE>],
    _Tensor
);

macro_rules! impl_bin_ops_basic {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
        impl<A, B, const DEVICE: usize> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where
        A: CommonBounds + NormalOut<B>,
        B: CommonBounds,
        <A as NormalOut<B>>::Output: CommonBounds,
        <A as NormalOut<B>>::Output: Cast<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
    {
        type Output = Tensor<NormalType<A, B>, Cpu, DEVICE>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = Tensor<NormalType<A, B>, Cpu, DEVICE>;
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.add_(rhs.inner.as_ref(), out.borrow().inner.as_ref())?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.sub_(rhs.inner.as_ref(), out.borrow().inner.as_ref())?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.mul_(rhs.inner.as_ref(), out.borrow().inner.as_ref())?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.rem_(rhs.inner.as_ref(), out.borrow().inner.as_ref())?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn pow(&self, rhs: $($rhs)*) -> std::result::Result<Self::Output, TensorError> {
            Ok(self.inner.pow(rhs.inner.as_ref())?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn pow_<U>(&self, rhs: $($rhs)*, out: U) -> std::result::Result<Self::Output, TensorError>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.pow_(rhs.inner.as_ref(), out.borrow().inner.as_ref())?.into())
        }
    }
    };
}

impl_bin_ops_basic!([Tensor<A, Cpu, DEVICE>], [&Tensor<B, Cpu, DEVICE>], Tensor);
impl_bin_ops_basic!([Tensor<A, Cpu, DEVICE>], [Tensor<B, Cpu, DEVICE>], Tensor);
impl_bin_ops_basic!([&Tensor<A, Cpu, DEVICE>], [&Tensor<B, Cpu, DEVICE>], Tensor);
impl_bin_ops_basic!([&Tensor<A, Cpu, DEVICE>], [Tensor<B, Cpu, DEVICE>], Tensor);
