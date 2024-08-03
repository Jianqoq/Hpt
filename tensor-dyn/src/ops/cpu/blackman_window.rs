use std::ops::{ Add, Div, Mul, Sub };

use rayon::iter::{ IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator };
use tensor_traits::CommonBounds;
use tensor_types::{
    convertion::FromScalar,
    dtype::FloatConst,
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};
use tensor_traits::TensorCreator;
use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T> +
            Add<Output = T>,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    pub fn blackman_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<T>> {
        let a0: T = (0.42).into_scalar();
        let a1: T = (0.5).into_scalar();
        let a2: T = (0.08).into_scalar();
        let length_usize = (if periodic { window_length } else { window_length - 1 }) as i64;
        let length: T = length_usize.into_scalar();
        let ret = _Tensor::<T>::empty(&[length_usize])?;
        ret.as_raw_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, x)| {
                let idx: T = idx.into_scalar();
                *x =
                    a0 -
                    a1 * ((T::TWOPI * idx) / length)._cos() +
                    a2 * ((T::FOURPI * idx) / length)._cos();
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
            FloatConst +
            Mul<Output = T> +
            Div<Output = T> +
            Sub<Output = T> +
            FloatOut<Output = T> +
            Add<Output = T>,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    pub fn blackman_window(window_length: i64, periodic: bool) -> anyhow::Result<Tensor<T>> {
        Ok(_Tensor::<T>::blackman_window(window_length, periodic)?.into())
    }
}
