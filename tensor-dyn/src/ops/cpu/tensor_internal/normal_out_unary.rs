use std::borrow::Borrow;
use tensor_common::err_handler::ErrHandler;
use tensor_traits::{ CommonBounds, NormalUaryOps, TensorLike };
use tensor_types::{ traits::VecTrait, type_promote::{ NormalOut, NormalOutUnary } };

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

    fn floor(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._floor(),
            |x| x._floor(),
            None::<Self::Output>
        )
    }

    fn floor_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._floor(),
            |x| x._floor(),
            Some(out)
        )
    }

    fn square(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._square(),
            |x| x._square(),
            None::<Self::Output>
        )
    }

    fn square_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._square(),
            |x| x._square(),
            Some(out)
        )
    }

    fn abs(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._abs(),
            |x| x._abs(),
            None::<Self::Output>
        )
    }

    fn abs_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._abs(),
            |x| x._abs(),
            Some(out)
        )
    }

    fn ceil(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._ceil(),
            |x| x._ceil(),
            None::<Self::Output>
        )
    }
    fn ceil_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._ceil(),
            |x| x._ceil(),
            Some(out)
        )
    }

    fn sign(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._signum(),
            |x| x._signum(),
            None::<Self::Output>
        )
    }
    fn sign_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._signum(),
            |x| x._signum(),
            Some(out)
        )
    }
    fn clamp(
        &self,
        min: NormalType<T>,
        max: NormalType<T>
    ) -> std::result::Result<Self::Output, ErrHandler> {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        uary_fn_with_out_simd(
            self,
            |x| x._clamp(min_vec, max_vec),
            |x| <T as NormalOut<T>>::_clamp(x, min, max),
            None::<Self::Output>
        )
    }
    fn clamp_<U>(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
        out: U
    ) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        uary_fn_with_out_simd(
            self,
            |x| x._clamp(min_vec, max_vec),
            |x| <T as NormalOut<T>>::_clamp(x, min, max),
            Some(out)
        )
    }
    fn round(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._round(),
            |x| x._round(),
            None::<Self::Output>
        )
    }
    fn round_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._round(),
            |x| x._round(),
            Some(out)
        )
    }

    fn neg(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._neg(),
            |x| x._neg(),
            None::<Self::Output>
        )
    }

    fn neg_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._neg(),
            |x| x._neg(),
            Some(out)
        )
    }

    fn relu(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu(),
            |x| x._relu(),
            None::<Self::Output>
        )
    }

    fn relu_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu(),
            |x| x._relu(),
            Some(out)
        )
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            None::<Self::Output>
        )
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> std::result::Result<Self::Output, ErrHandler>
        where U: Borrow<Self::InplaceOutput>
    {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            Some(out)
        )
    }

    fn relu6(&self) -> std::result::Result<Self::Output, ErrHandler> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            None::<Self::Output>
        )
    }

    fn relu6_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler> where U: Borrow<Self::InplaceOutput> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            Some(out)
        )
    }
}
