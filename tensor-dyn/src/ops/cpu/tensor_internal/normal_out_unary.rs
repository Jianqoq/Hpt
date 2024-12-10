use std::borrow::Borrow;
use tensor_traits::{ CommonBounds, NormalUaryOps, TensorLike };
use tensor_types::{ traits::Init, type_promote::{ NormalOut, NormalOutUnary } };

use crate::{ ops::cpu::unary::uary_fn_with_out_simd, tensor_base::_Tensor };

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T> NormalUaryOps
    for _Tensor<T>
    where
        T: CommonBounds,
        T::Vec: NormalOutUnary<Base = NormalType<T>>,
        T: NormalOutUnary<Base = NormalType<T>>,
        _Tensor<NormalType<T>>: TensorLike<NormalType<T>>
{
    type Output = _Tensor<NormalType<T>>;

    type InplaceOutput = _Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._floor(),
            |x| x._floor(),
            None::<_Tensor<NormalType<T>>>
        )
    }

    fn floor_<U>(&self, out: U) -> anyhow::Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._floor(),
            |x| x._floor(),
            Some(out)
        )
    }

    fn square(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._square(),
            |x| x._square(),
            None::<_Tensor<NormalType<T>>>
        )
    }

    fn square_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._square(),
            |x| x._square(),
            Some(out)
        )
    }

    fn abs(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            |x| x._abs(),
            |x| x._abs(),
            None::<Self::Output>
        )
    }

    fn abs_<U>(&self, out: U) -> anyhow::Result<Self> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._abs(),
            |x| x._abs(),
            Some(out)
        )
    }

    fn ceil(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._ceil(),
            |x| x._ceil(),
            None::<_Tensor<NormalType<T>>>
        )
    }
    fn ceil_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._ceil(),
            |x| x._ceil(),
            Some(out)
        )
    }

    fn sign(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sign(),
            |x| x._sign(),
            None::<_Tensor<NormalType<T>>>
        )
    }
    fn sign_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._sign(),
            |x| x._sign(),
            Some(out)
        )
    }
    fn clamp(
        &self,
        min: NormalType<T>,
        max: NormalType<T>
    ) -> anyhow::Result<_Tensor<NormalType<T>>> {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        uary_fn_with_out_simd(
            self,
            |x| x._clip(min_vec, max_vec),
            |x| <T as NormalOut<T>>::_clip(x, min, max),
            None::<_Tensor<NormalType<T>>>
        )
    }
    fn clamp_<U>(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
        out: U
    ) -> anyhow::Result<_Tensor<NormalType<T>>>
        where U: Borrow<Self::InplaceOutput>
    {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        uary_fn_with_out_simd(
            self,
            |x| x._clip(min_vec, max_vec),
            |x| <T as NormalOut<T>>::_clip(x, min, max),
            Some(out)
        )
    }
    fn round(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._round(),
            |x| x._round(),
            None::<_Tensor<NormalType<T>>>
        )
    }
    fn round_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._round(),
            |x| x._round(),
            Some(out)
        )
    }

    fn neg(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._neg(),
            |x| x._neg(),
            None::<_Tensor<NormalType<T>>>
        )
    }

    fn neg_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._neg(),
            |x| x._neg(),
            Some(out)
        )
    }

    fn relu(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu(),
            |x| x._relu(),
            None::<_Tensor<NormalType<T>>>
        )
    }

    fn relu_<U>(&self, out: U) -> anyhow::Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu(),
            |x| x._relu(),
            Some(out)
        )
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            None::<_Tensor<NormalType<T>>>
        )
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            Some(out)
        )
    }

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            None::<_Tensor<NormalType<T>>>
        )
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            Some(out)
        )
    }
}
