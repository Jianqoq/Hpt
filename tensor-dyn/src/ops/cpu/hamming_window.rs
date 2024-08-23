use std::ops::{ Div, Mul, Sub };

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::FromScalar,
    dtype::{ FloatConst, TypeCommon },
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};

use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T>,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hamming_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<T>> {
        let alpha: T = (0.54).into_scalar();
        let beta: T = (0.46).into_scalar();
        let length_usize = (if periodic { window_length } else { window_length - 1 }) as i64;
        let length: T = length_usize.into_scalar();
        let ret = _Tensor::<T>::empty(&[length_usize])?;
        ret.as_raw_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, x)| {
                *x = alpha - beta * ((T::TWO * T::PI * idx.into_scalar()) / length)._cos();
            });
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T>,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hamming_window(window_length: i64, periodic: bool) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hamming_window(window_length, periodic)?.into()))
    }
}
