use std::borrow::Borrow;
use tensor_traits::{ CommonBounds, NormalUaryOps, TensorLike };
use tensor_types::{
    traits::Init,
    into_scalar::IntoScalar,
    type_promote::{ NormalOut, NormalOutUnary },
};

use crate::{ ops::cpu::unary::uary_fn_with_out_simd, tensor_base::_Tensor };

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T> NormalUaryOps
    for _Tensor<T>
    where
        T: CommonBounds + IntoScalar<T>,
        NormalType<T>: CommonBounds,
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
    fn clip(
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
    fn clip_<U>(
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
}
