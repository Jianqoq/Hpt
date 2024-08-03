use std::ops::{ Div, Mul, Sub };

use num::Float;
use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::{ FloatConst, TypeCommon },
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};

use crate::{tensor::Tensor, tensor_base::_Tensor};

impl<T> _Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            Convertor +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T> +
            Float,
        usize: IntoScalar<T>
{
    pub fn hann_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<T>> {
        let length_usize = (if periodic { window_length } else { window_length - 1 }) as usize;
        let length: T = length_usize.into_scalar();
        let ret = _Tensor::<T>::linspace(T::ZERO, length, length_usize, false)?;
        ret.as_raw_mut()
            .iter_mut()
            .for_each(|x| {
                *x = T::HALF * (T::ONE - ((T::TWO * T::PI * *x) / length)._cos());
            });
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            Convertor +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T> +
            Float,
        usize: IntoScalar<T>
{
    pub fn hann_window(window_length: i64, periodic: bool) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hann_window(window_length, periodic)?.into()))
    }
}