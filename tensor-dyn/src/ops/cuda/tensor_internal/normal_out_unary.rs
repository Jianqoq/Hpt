use crate::{
    ops::cuda::{cuda_utils::get_module_name_1, unary::uary_fn_with_out_simd},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::DeviceRepr;
use std::borrow::Borrow;
use tensor_traits::{CommonBounds, NormalUaryOps, TensorLike};
use tensor_types::cuda_types::scalar::Scalar;
use tensor_types::type_promote::{NormalOut, NormalOutUnary};

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T, const DEVICE_ID: usize> NormalUaryOps for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr,
    T::Vec: NormalOutUnary<Base = NormalType<T>>,
    T: NormalOutUnary<Base = NormalType<T>>,
    _Tensor<NormalType<T>, Cuda, DEVICE_ID>: TensorLike<NormalType<T>>,
    Scalar<T>:
        NormalOutUnary<Base = Scalar<NormalType<T>>> + NormalOut<Output = Scalar<NormalType<T>>>,
{
    type Output = _Tensor<NormalType<T>, Cuda, DEVICE_ID>;

    type InplaceOutput = _Tensor<NormalType<T>, Cuda, DEVICE_ID>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("floor", self),
            |out, x| out.assign(x._floor()),
            None::<Self::Output>,
        )
    }

    fn floor_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("floor", self),
            |out, x| out.assign(x._floor()),
            Some(out),
        )
    }

    fn square(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("square", self),
            |out, x| out.assign(x._square()),
            None::<Self::Output>,
        )
    }

    fn square_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("square", self),
            |out, x| out.assign(x._square()),
            Some(out),
        )
    }

    fn abs(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("abs", self),
            |out, x| out.assign(x._abs()),
            None::<Self::Output>,
        )
    }

    fn abs_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("abs", self),
            |out, x| out.assign(x._abs()),
            Some(out),
        )
    }

    fn ceil(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ceil", self),
            |out, x| out.assign(x._ceil()),
            None::<Self::Output>,
        )
    }
    fn ceil_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ceil", self),
            |out, x| out.assign(x._ceil()),
            Some(out),
        )
    }

    fn sign(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sign", self),
            |out, x| out.assign(x._sign()),
            None::<Self::Output>,
        )
    }
    fn sign_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sign", self),
            |out, x| out.assign(x._sign()),
            Some(out),
        )
    }
    fn clamp(&self, min: NormalType<T>, max: NormalType<T>) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("clamp", self),
            |out, x| {
                let min_scalar = Scalar::new(min);
                let max_scalar = Scalar::new(max);
                out.assign(x._clip(min_scalar, max_scalar))
            },
            None::<Self::Output>,
        )
    }
    fn clamp_<U>(&self, min: NormalType<T>, max: NormalType<T>, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("clamp", self),
            |out, x| {
                let min_scalar = Scalar::new(min);
                let max_scalar = Scalar::new(max);
                out.assign(x._clip(min_scalar, max_scalar))
            },
            Some(out),
        )
    }
    fn round(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("round", self),
            |out, x| out.assign(x._round()),
            None::<Self::Output>,
        )
    }
    fn round_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("round", self),
            |out, x| out.assign(x._round()),
            Some(out),
        )
    }

    fn neg(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("neg", self),
            |out, x| out.assign(x._neg()),
            None::<Self::Output>,
        )
    }

    fn neg_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("neg", self),
            |out, x| out.assign(x._neg()),
            Some(out),
        )
    }

    fn relu(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("relu", self),
            |out, x| out.assign(x._relu()),
            None::<Self::Output>,
        )
    }

    fn relu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("relu", self),
            |out, x| out.assign(x._relu()),
            Some(out),
        )
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
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

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
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

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| out.assign(x._relu6()),
            None::<Self::Output>,
        )
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| out.assign(x._relu6()),
            Some(out),
        )
    }
}
