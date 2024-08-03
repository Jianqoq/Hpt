use std::ops::{ Div, Mul, Sub };

use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};

use crate::tensor_base::_Tensor;

impl<T> _Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            Convertor +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            tensor_types::dtype::FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T>,
        usize: IntoScalar<T>
{
    pub fn hamming_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<T>> {
        let alpha: T = (0.54).into_scalar();
        let beta: T = (0.46).into_scalar();
        let length: T = (
            (if periodic { window_length } else { window_length - 1 }) as f64
        ).into_scalar();
        let ret = _Tensor::<T>::arange(T::ZERO, length)?;
        ret.as_raw_mut()
            .iter_mut()
            .for_each(|x| {
                *x = alpha - beta * ((T::TWO * T::PI * *x) / length)._cos();
            });
        Ok(ret)
    }
}
