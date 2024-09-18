use crate::{tensor::Tensor, tensor_base::_Tensor};
use std::borrow::{Borrow, BorrowMut};
use tensor_traits::{
    ops::binary::{Matmul, NormalBinOps},
    tensor::CommonBounds,
};
use tensor_types::{into_scalar::IntoScalar, type_promote::NormalOut};

use super::{
    binary_normal::binary_fn_with_out,
    matmul::{matmul_no_out, matmul_with_out},
};

pub(crate) type NormalType<A, B> = <A as NormalOut<B>>::Output;

macro_rules! impl_bin_ops {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
    impl<A, B> NormalBinOps<$($rhs)*>
        for $($lhs)*
        where A: CommonBounds + NormalOut<B>, B: CommonBounds, <A as NormalOut<B>>::Output: CommonBounds
    {
        type Output = $output<NormalType<A, B>>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>>;

        #[cfg_attr(feature = "track_caller", track_caller)]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._add(b), out)
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out)
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out)
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out)
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
        where A: CommonBounds + NormalOut<B>, B: CommonBounds, <A as NormalOut<B>>::Output: CommonBounds
    {
        type Output = Tensor<NormalType<A, B>>;
        type OutputMeta = NormalType<A, B>;
        type InplaceOutput = _Tensor<NormalType<A, B>>;
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._add(b), out)?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out)?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out)?.into())
        }
        #[cfg_attr(feature = "track_caller", track_caller)]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: Borrow<Self::InplaceOutput>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out)?.into())
        }
    }
    };
}

impl_bin_ops_basic!([Tensor<A>], [&Tensor<B>], Tensor);
impl_bin_ops_basic!([Tensor<A>], [Tensor<B>], Tensor);
impl_bin_ops_basic!([&Tensor<A>], [&Tensor<B>], Tensor);
impl_bin_ops_basic!([&Tensor<A>], [Tensor<B>], Tensor);

impl<A, B> Matmul<_Tensor<B>> for _Tensor<A>
where
    A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
    B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn matmul(&self, rhs: _Tensor<B>) -> anyhow::Result<Self::Output> {
        matmul_no_out(self, &rhs)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn matmul_<U>(&self, rhs: _Tensor<B>, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, &rhs, out)
    }
}

impl<A, B> Matmul<&_Tensor<B>> for _Tensor<A>
where
    A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
    B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn matmul(&self, rhs: &_Tensor<B>) -> anyhow::Result<Self::Output> {
        matmul_no_out(self, rhs)
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn matmul_<U>(&self, rhs: &_Tensor<B>, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, rhs, out)
    }
}
