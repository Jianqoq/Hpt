use tensor_traits::{
    ops::binary::{ Matmul, NormalBinOps },
    tensor::{ CommonBounds, TensorInfo, TensorLike },
};
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };

use crate::tensor::_Tensor;

use super::{ binary_funcs_normal::binary_fn_with_out, matmul::{ matmul_no_out, matmul_with_out } };

type NormalType<T> = <T as NormalOut>::Output;

impl<T> NormalBinOps for _Tensor<T> where T: NormalOut + CommonBounds, NormalType<T>: CommonBounds {
    type Output = _Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    type InplaceOutput = _Tensor<NormalType<T>>;

    fn add_<U>(&self, rhs: Self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        binary_fn_with_out(self, &rhs, |a, b| a._add(b), out)
    }

    fn sub_<U>(&self, rhs: Self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        binary_fn_with_out(self, &rhs, |a, b| a._sub(b), out)
    }

    fn mul_<U>(&self, rhs: Self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        binary_fn_with_out(self, &rhs, |a, b| a._mul(b), out)
    }

    fn rem_<U>(&self, rhs: Self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        binary_fn_with_out(self, &rhs, |a, b| a._rem(b), out)
    }

    fn convolve(&self, rhs: Self) -> anyhow::Result<Self::Output> {
        todo!()
    }
}

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
