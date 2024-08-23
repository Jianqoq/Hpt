use tensor_traits::{
    ops::binary::{ Matmul, NormalBinOps },
    tensor::{ CommonBounds, TensorInfo, TensorLike },
};
use std::panic::Location;
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };

use crate::{ tensor::Tensor, tensor_base::_Tensor };

use super::{ binary_normal::binary_fn_with_out, matmul::{ matmul_no_out, matmul_with_out } };

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

        #[track_caller]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._add(b), out, Location::caller())
        }
        #[track_caller]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out, Location::caller())
        }
        #[track_caller]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out, Location::caller())
        }
        #[track_caller]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out, Location::caller())
        }
        fn convolve(&self, _: $($rhs)*) -> anyhow::Result<Self::Output> {
            todo!()
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
        #[track_caller]
        fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._add(b), out, Location::caller())?.into())
        }
        #[track_caller]
        fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out, Location::caller())?.into())
        }
        #[track_caller]
        fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out, Location::caller())?.into())
        }
        #[track_caller]
        fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
            where
                U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                    TensorInfo<Self::OutputMeta>
        {
            Ok(binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out, Location::caller())?.into())
        }
        fn convolve(&self, _: $($rhs)*) -> anyhow::Result<Self::Output> {
            todo!()
        }
    }
    };
}

impl_bin_ops_basic!([Tensor<A>], [&Tensor<B>], Tensor);
impl_bin_ops_basic!([Tensor<A>], [Tensor<B>], Tensor);
impl_bin_ops_basic!([&Tensor<A>], [&Tensor<B>], Tensor);
impl_bin_ops_basic!([&Tensor<A>], [Tensor<B>], Tensor);

impl<A, B> Matmul<_Tensor<B>>
    for _Tensor<A>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    #[track_caller]
    fn matmul(&self, rhs: _Tensor<B>) -> anyhow::Result<Self::Output> {
        matmul_no_out(self, &rhs, Location::caller())
    }
    #[track_caller]
    fn matmul_<U>(&self, rhs: _Tensor<B>, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        matmul_with_out(self, &rhs, out, Location::caller())
    }
}

impl<A, B> Matmul<&_Tensor<B>>
    for _Tensor<A>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    #[track_caller]
    fn matmul(&self, rhs: &_Tensor<B>) -> anyhow::Result<Self::Output> {
        matmul_no_out(self, rhs, Location::caller())
    }

    #[track_caller]
    fn matmul_<U>(&self, rhs: &_Tensor<B>, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        matmul_with_out(self, rhs, out, Location::caller())
    }
}
