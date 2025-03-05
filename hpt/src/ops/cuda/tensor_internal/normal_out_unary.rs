use crate::{
    ops::cuda::{cuda_utils::get_module_name_1, utils::unary::unary::uary_fn_with_out_simd},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, NormalUaryOps, TensorLike};
use hpt_types::type_promote::{NormalOut, NormalOutUnary};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType};
use std::borrow::BorrowMut;

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T, const DEVICE: usize, Al> NormalUaryOps for _Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds + DeviceRepr + CudaType + NormalOutUnary,
    _Tensor<NormalType<T>, Cuda, DEVICE, Al>: TensorLike<NormalType<T>>,
    Scalar<T>: NormalOutUnary + NormalOut<Output = Scalar<NormalType<T>>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<NormalType<T>, Cuda, DEVICE, Al>;

    type InplaceOutput = _Tensor<NormalType<T>, Cuda, DEVICE, Al>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("floor", self),
            |out, x| out.assign(x._floor()),
            None::<Self::Output>,
        )
    }

    fn floor_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("floor", self),
            |out, x| out.assign(x._floor()),
            Some(out),
        )
    }

    fn square(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("square", self),
            |out, x| out.assign(x._square()),
            None::<Self::Output>,
        )
    }

    fn square_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("square", self),
            |out, x| out.assign(x._square()),
            Some(out),
        )
    }

    fn abs(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("abs", self),
            |out, x| out.assign(x._abs()),
            None::<Self::Output>,
        )
    }

    fn abs_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("abs", self),
            |out, x| out.assign(x._abs()),
            Some(out),
        )
    }

    fn ceil(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ceil", self),
            |out, x| out.assign(x._ceil()),
            None::<Self::Output>,
        )
    }
    fn ceil_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ceil", self),
            |out, x| out.assign(x._ceil()),
            Some(out),
        )
    }

    fn sign(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sign", self),
            |out, x| out.assign(x._signum()),
            None::<Self::Output>,
        )
    }
    fn sign_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sign", self),
            |out, x| out.assign(x._signum()),
            Some(out),
        )
    }
    fn clamp(&self, min: NormalType<T>, max: NormalType<T>) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("clamp", self),
            |out, x| {
                let min_scalar = Scalar::new(min);
                let max_scalar = Scalar::new(max);
                out.assign(x._clamp(min_scalar, max_scalar))
            },
            None::<Self::Output>,
        )
    }
    fn clamp_<U>(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("clamp", self),
            |out, x| {
                let min_scalar = Scalar::new(min);
                let max_scalar = Scalar::new(max);
                out.assign(x._clamp(min_scalar, max_scalar))
            },
            Some(out),
        )
    }
    fn round(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("round", self),
            |out, x| out.assign(x._round()),
            None::<Self::Output>,
        )
    }
    fn round_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("round", self),
            |out, x| out.assign(x._round()),
            Some(out),
        )
    }

    fn neg(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("neg", self),
            |out, x| out.assign(x._neg()),
            None::<Self::Output>,
        )
    }

    fn neg_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("neg", self),
            |out, x| out.assign(x._neg()),
            Some(out),
        )
    }

    fn relu(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("relu", self),
            |out, x| out.assign(x._relu()),
            None::<Self::Output>,
        )
    }

    fn relu_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("relu", self),
            |out, x| out.assign(x._relu()),
            Some(out),
        )
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| {
                let alpha_scalar = Scalar::new(alpha);
                out.assign(x._leaky_relu(alpha_scalar))
            },
            None::<Self::Output>,
        )
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| {
                let alpha_scalar = Scalar::new(alpha);
                out.assign(x._leaky_relu(alpha_scalar))
            },
            Some(out),
        )
    }

    fn relu6(&self) -> Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| out.assign(x._relu6()),
            None::<Self::Output>,
        )
    }

    fn relu6_<U>(&self, out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| out.assign(x._relu6()),
            Some(out),
        )
    }
}
