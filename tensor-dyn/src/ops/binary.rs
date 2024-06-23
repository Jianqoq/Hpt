use tensor_traits::{
    ops::binary::{ Matmul, NormalBinOps },
    tensor::{ CommonBounds, TensorInfo, TensorLike },
};
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };

use crate::{ tensor::Tensor, tensor_base::_Tensor };

use super::{ binary_normal::binary_fn_with_out, matmul::{ matmul_no_out, matmul_with_out } };

type NormalType<T> = <T as NormalOut>::Output;

macro_rules! impl_bin_ops {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
        impl<T> NormalBinOps<$($rhs)*> for $($lhs)* where T: NormalOut + CommonBounds, NormalType<T>: CommonBounds {
            type Output = $output<NormalType<T>>;
        
            type OutputMeta = NormalType<T>;
        
            type InplaceOutput = $output<NormalType<T>>;
        
            fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                binary_fn_with_out(self, &rhs, |a, b| a._add(b), out)
            }
        
            fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out)
            }
        
            fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out)
            }
        
            fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out)
            }
        
            fn convolve(&self, _: $($rhs)*) -> anyhow::Result<Self::Output> {
                todo!()
            }
        }
    };
}

impl_bin_ops!([_Tensor<T>], [_Tensor<T>], _Tensor);
impl_bin_ops!([_Tensor<T>], [&_Tensor<T>], _Tensor);
impl_bin_ops!([&_Tensor<T>], [&_Tensor<T>], _Tensor);
impl_bin_ops!([&_Tensor<T>], [_Tensor<T>], _Tensor);

macro_rules! impl_bin_ops_basic {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $output:ident
    ) => {
        impl<T> NormalBinOps<$($rhs)*> for $($lhs)* where T: NormalOut + CommonBounds, NormalType<T>: CommonBounds {
            type Output = $output<NormalType<T>>;
        
            type OutputMeta = NormalType<T>;
        
            type InplaceOutput = _Tensor<NormalType<T>>;
        
            fn add_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                Ok(binary_fn_with_out(self, &rhs, |a, b| a._add(b), out)?.into())
            }
        
            fn sub_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                Ok(binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out)?.into())
            }
        
            fn mul_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                Ok(binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out)?.into())
            }
        
            fn rem_<U>(&self, rhs: $($rhs)*, out: U) -> anyhow::Result<Self::Output>
                where
                    U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                        TensorInfo<Self::OutputMeta>
            {
                Ok(binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out)?.into())
            }
        
            fn convolve(&self, _: $($rhs)*) -> anyhow::Result<Self::Output> {
                todo!()
            }
        }
    };
}

impl_bin_ops_basic!([Tensor<T>], [Tensor<T>], Tensor);
impl_bin_ops_basic!([Tensor<T>], [&Tensor<T>], Tensor);
impl_bin_ops_basic!([&Tensor<T>], [&Tensor<T>], Tensor);
impl_bin_ops_basic!([&Tensor<T>], [Tensor<T>], Tensor);

impl<T> Matmul
    for _Tensor<T>
    where T: NormalOut + CommonBounds + IntoScalar<NormalType<T>>, NormalType<T>: CommonBounds
{
    type Output = _Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    type InplaceOutput = _Tensor<NormalType<T>>;

    fn matmul(&self, rhs: Self) -> anyhow::Result<Self::Output> {
        matmul_no_out(self, &rhs)
    }

    fn matmul_<U>(&self, rhs: Self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        matmul_with_out(self, &rhs, out)
    }
}
