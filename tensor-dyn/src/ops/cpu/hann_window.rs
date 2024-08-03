use std::ops::{ Div, Mul, Sub };

use num::Float;
use rayon::iter::{ IndexedParallelIterator, IntoParallelIterator, ParallelIterator };
use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::{ FloatConst, TypeCommon },
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};

use crate::{ tensor::Tensor, tensor_base::_Tensor };

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
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    pub fn hann_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<T>> {
        let length_i64 = (if periodic { window_length } else { window_length - 1 }) as i64;
        let length: T = length_i64.into_scalar();
        let data = _Tensor::empty(&[length_i64])?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                let i = T::ZERO._add(i.into_scalar()._mul(T::ONE));
                *x = T::HALF * (T::ONE - ((T::TWO * T::PI * i) / length)._cos());
            });
        Ok(data)
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
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    pub fn hann_window(window_length: i64, periodic: bool) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hann_window(window_length, periodic)?.into()))
    }
}
