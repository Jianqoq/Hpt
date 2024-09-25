use crate::ops::cpu::binary_normal::binary_fn_with_out_simd;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use std::borrow::Borrow;
use tensor_traits::{ops::binary::NormalBinOps, tensor::CommonBounds};
use tensor_types::dtype::TypeCommon;
use tensor_types::{into_scalar::IntoScalar, type_promote::NormalOut};

/// a type alias for the output type of the binary operations of `A` and `B`
pub(crate) type NormalType<A, B> = <A as NormalOut<B>>::Output;

macro_rules! impl_bin_ops {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
    impl<A, B> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where
        A: CommonBounds + NormalOut<B>,
        B: CommonBounds,
        <A as NormalOut<B>>::Output: CommonBounds,
        <A as NormalOut<B>>::Output: IntoScalar<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
    {
        type Output = $output<NormalType<A, B>>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>>;

        #[cfg_attr(feature = "track_caller", track_caller)]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._add(b), |a, b| a._add(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._sub(b), |a, b| a._sub(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._mul(b), |a, b| a._mul(b), Some(out))
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out_simd(self, &rhs, |a, b| a._rem(b), |a, b| a._rem(b), Some(out))
        }
    }
    };
}

impl_bin_ops!([_Tensor<A>], [&_Tensor<B>], _Tensor);
impl_bin_ops!([_Tensor<A>], [_Tensor<B>], _Tensor);
impl_bin_ops!([&_Tensor<A>], [&_Tensor<B>], _Tensor);
impl_bin_ops!([&_Tensor<A>], [_Tensor<B>], _Tensor);

macro_rules! impl_bin_ops_basic {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
        impl<A, B> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where
        A: CommonBounds + NormalOut<B>,
        B: CommonBounds,
        <A as NormalOut<B>>::Output: CommonBounds,
        <A as NormalOut<B>>::Output: IntoScalar<<A as NormalOut<B>>::Output>,
        A::Vec: NormalOut<B::Vec, Output = <<A as NormalOut<B>>::Output as TypeCommon>::Vec>,
    {
        type Output = Tensor<NormalType<A, B>>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>>;
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.add_(rhs.inner.as_ref(), out)?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.sub_(rhs.inner.as_ref(), out)?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.mul_(rhs.inner.as_ref(), out)?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        #[inline]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(self.inner.rem_(rhs.inner.as_ref(), out)?.into())
        }
    }
    };
}

impl_bin_ops_basic!([Tensor<A>], [&Tensor<B>], Tensor);
impl_bin_ops_basic!([Tensor<A>], [Tensor<B>], Tensor);
impl_bin_ops_basic!([&Tensor<A>], [&Tensor<B>], Tensor);
impl_bin_ops_basic!([&Tensor<A>], [Tensor<B>], Tensor);
