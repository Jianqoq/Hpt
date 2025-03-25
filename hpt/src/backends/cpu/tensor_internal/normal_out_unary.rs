use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::error::base::TensorError;
use hpt_iterator::TensorIterator;
use hpt_traits::{
    ops::unary::{Contiguous, NormalUaryOps},
    tensor::{CommonBounds, TensorLike},
};
use hpt_types::{
    traits::VecTrait,
    type_promote::{NormalOut, NormalOutUnary},
};
use std::borrow::Borrow;

use crate::{backends::cpu::utils::unary::unary::unary_fn_with_out, tensor_base::_Tensor};

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T, A2, const DEVICE: usize> NormalUaryOps for _Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds,
    T::Vec: NormalOutUnary,
    _Tensor<NormalType<T>, Cpu, DEVICE, A2>: TensorLike<NormalType<T>>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<NormalType<T>, Cpu, DEVICE, A2>;

    type InplaceOutput = _Tensor<NormalType<T>, Cpu, DEVICE, A2>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._floor(), |x| x._floor(), None::<Self::Output>)
    }

    fn floor_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._floor(), |x| x._floor(), Some(out))
    }

    fn square(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._square(), |x| x._square(), None::<Self::Output>)
    }

    fn square_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._square(), |x| x._square(), Some(out))
    }

    fn abs(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._abs(), |x| x._abs(), None::<Self::Output>)
    }

    fn abs_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._abs(), |x| x._abs(), Some(out))
    }

    fn ceil(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._ceil(), |x| x._ceil(), None::<Self::Output>)
    }
    fn ceil_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._ceil(), |x| x._ceil(), Some(out))
    }

    fn sign(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._signum(), |x| x._signum(), None::<Self::Output>)
    }
    fn sign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._signum(), |x| x._signum(), Some(out))
    }
    fn clamp(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
    ) -> std::result::Result<Self::Output, TensorError> {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        unary_fn_with_out(
            self,
            |x| x._clamp(min_vec, max_vec),
            |x| <T as NormalOut<T>>::_clamp(x, min, max),
            None::<Self::Output>,
        )
    }
    fn clamp_<U>(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        unary_fn_with_out(
            self,
            |x| x._clamp(min_vec, max_vec),
            |x| <T as NormalOut<T>>::_clamp(x, min, max),
            Some(out),
        )
    }
    fn round(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._round(), |x| x._round(), None::<Self::Output>)
    }
    fn round_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._round(), |x| x._round(), Some(out))
    }

    fn neg(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._neg(), |x| x._neg(), None::<Self::Output>)
    }

    fn neg_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._neg(), |x| x._neg(), Some(out))
    }

    fn relu(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._relu(), |x| x._relu(), None::<Self::Output>)
    }

    fn relu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._relu(), |x| x._relu(), Some(out))
    }

    fn leaky_relu(
        &self,
        alpha: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha_vec = T::Vec::splat(alpha);
        unary_fn_with_out(
            self,
            |x| x._leaky_relu(alpha_vec),
            |x| x._leaky_relu(alpha),
            None::<Self::Output>,
        )
    }

    fn leaky_relu_<U>(
        &self,
        alpha: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha_vec = T::Vec::splat(alpha);
        unary_fn_with_out(
            self,
            |x| x._leaky_relu(alpha_vec),
            |x| x._leaky_relu(alpha),
            Some(out),
        )
    }

    fn relu6(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._relu6(), |x| x._relu6(), None::<Self::Output>)
    }

    fn relu6_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._relu6(), |x| x._relu6(), Some(out))
    }
}

impl<T, const DEVICE: usize, Al> Contiguous for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn contiguous(&self) -> Result<Self, TensorError> {
        Ok(self.par_iter().strided_map(|(res, x)| *res = x).collect())
    }
}
