use rayon::iter::{ IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator };
use tensor_traits::{CommonBounds, TensorLike};
use tensor_types::{
    convertion::FromScalar,
    dtype::FloatConst,
    into_scalar::IntoScalar,
    type_promote::{ FloatOutBinary, FloatOutUnary, NormalOut },
};
use tensor_traits::TensorCreator;
use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        T: CommonBounds +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            FloatConst +
            FloatOutUnary +
            FloatOutBinary +
            FloatOutBinary<<T as FloatOutUnary>::Output, Output = <T as FloatOutBinary>::Output> +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: FloatOutUnary<Output = <T as FloatOutBinary>::Output> +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            CommonBounds,
        usize: IntoScalar<<T as FloatOutBinary>::Output>,
        i64: IntoScalar<<T as FloatOutBinary>::Output>
{
    pub fn blackman_window(
        window_length: i64,
        periodic: bool
    ) -> anyhow::Result<_Tensor<<T as FloatOutBinary>::Output>> {
        let a0: <T as FloatOutBinary>::Output = 0.42.into_scalar();
        let a1: <T as FloatOutBinary>::Output = 0.5.into_scalar();
        let a2: <T as FloatOutBinary>::Output = 0.08.into_scalar();
        let length_usize = if periodic { window_length } else { window_length - 1 };
        let length: <T as FloatOutBinary>::Output = length_usize.into_scalar();
        let mut ret = _Tensor::<<T as FloatOutBinary>::Output>::empty(&[length_usize])?;
        ret.as_raw_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, x)| {
                let idx: <T as FloatOutBinary>::Output = idx.into_scalar();
                let a = a1._mul(T::TWOPI._mul(idx)._div(length)._cos());
                let b = a2._mul(T::FOURPI._mul(idx)._div(length)._cos());
                *x = a0._sub(a)._add(b);
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
            FloatConst +
            FloatOutUnary +
            FloatOutBinary +
            FloatOutBinary<<T as FloatOutUnary>::Output, Output = <T as FloatOutBinary>::Output> +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: FloatOutUnary<Output = <T as FloatOutBinary>::Output> +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            CommonBounds,
        usize: IntoScalar<<T as FloatOutBinary>::Output>,
        i64: IntoScalar<<T as FloatOutBinary>::Output>
{
    pub fn blackman_window(
        window_length: i64,
        periodic: bool
    ) -> anyhow::Result<Tensor<<T as FloatOutBinary>::Output>> {
        Ok(_Tensor::<T>::blackman_window(window_length, periodic)?.into())
    }
}
