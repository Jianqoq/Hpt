use std::ops::{ Mul, Sub };

use rayon::iter::{ IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator };
use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::FromScalar,
    dtype::{ FloatConst, TypeCommon },
    into_scalar::IntoScalar,
    type_promote::{ FloatOutBinary, FloatOutUnary, NormalOut },
};

use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        T: CommonBounds +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            Mul<Output = T> +
            Sub<Output = T> +
            FloatOutBinary,
        <T as FloatOutBinary>::Output: CommonBounds +
            FloatOutUnary<Output = <T as FloatOutBinary>::Output> +
            Mul<Output = <T as FloatOutBinary>::Output> +
            Sub<Output = <T as FloatOutBinary>::Output>,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hamming_window(
        window_length: i64,
        periodic: bool
    ) -> anyhow::Result<_Tensor<<T as FloatOutBinary>::Output>> {
        let alpha: <T as FloatOutBinary>::Output = (0.54).into_scalar();
        let beta: <T as FloatOutBinary>::Output = (0.46).into_scalar();
        let length_usize = (if periodic { window_length } else { window_length - 1 }) as i64;
        let length: T = length_usize.into_scalar();
        let ret = _Tensor::<<T as FloatOutBinary>::Output>::empty(&[length_usize])?;
        ret.as_raw_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, x)| {
                *x = alpha - beta * (T::TWOPI * idx.into_scalar())._div(length)._cos();
            });
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        T: CommonBounds +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            Mul<Output = T> +
            Sub<Output = T> +
            FloatOutBinary,
        <T as FloatOutBinary>::Output: CommonBounds +
            FloatOutUnary<Output = <T as FloatOutBinary>::Output> +
            Mul<Output = <T as FloatOutBinary>::Output> +
            Sub<Output = <T as FloatOutBinary>::Output>,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hamming_window(
        window_length: i64,
        periodic: bool
    ) -> anyhow::Result<Tensor<<T as FloatOutBinary>::Output>> {
        Ok(Tensor::from(_Tensor::hamming_window(window_length, periodic)?.into()))
    }
}
