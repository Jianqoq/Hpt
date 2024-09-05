use rayon::iter::{ IndexedParallelIterator, ParallelIterator };
use tensor_traits::{ CommonBounds, TensorCreator };
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::{ FloatConst, TypeCommon },
    into_scalar::IntoScalar,
    type_promote::{ FloatOutBinary, FloatOutUnary, NormalOut },
};
use crate::{ backend::Cpu, tensor::Tensor, tensor_base::_Tensor };

#[cfg(not(feature = "simd"))]
use rayon::iter::IntoParallelIterator;

#[cfg(feature = "simd")]
use tensor_types::vectors::traits::*;

impl<T> _Tensor<T>
    where
        f64: IntoScalar<T>,
        T: CommonBounds +
            Convertor +
            NormalOut<T, Output = T> +
            FromScalar<T> +
            TypeCommon +
            FloatConst +
            FloatOutUnary +
            FloatOutBinary +
            FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: FloatOutUnary<Output = <T as FloatOutBinary>::Output> +
            NormalOut<Output = <T as FloatOutBinary>::Output> +
            FloatOutBinary<Output = <T as FloatOutBinary>::Output> +
            CommonBounds +
            FloatConst,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec> +
            FloatOutBinary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec> +
            FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>,
        usize: IntoScalar<T> + IntoScalar<<T as FloatOutBinary>::Output>,
        i64: IntoScalar<T> + IntoScalar<<T as FloatOutBinary>::Output>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hann_window(
        window_length: i64,
        periodic: bool
    ) -> anyhow::Result<_Tensor<<T as FloatOutBinary>::Output>> {
        let length_i64 = (if periodic { window_length } else { window_length - 1 }) as i64;
        let data = _Tensor::<<T as FloatOutBinary>::Output, Cpu>::empty(&[length_i64])?;

        #[cfg(feature = "simd")]
        {
            let per_thread_len = data.size() / rayon::current_num_threads();
            let per_thread_remain =
                per_thread_len % <<T as FloatOutBinary>::Output as TypeCommon>::Vec::SIZE;
            let per_thread_real_len = per_thread_len - per_thread_remain;
            let remain = if per_thread_real_len > 0 {
                data.size() % per_thread_real_len
            } else {
                data.size()
            };
            // value = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (length as f64 - 1.0)).cos();
            let one = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(
                <T as FloatOutBinary>::Output::ONE
            );
            let half = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(
                <T as FloatOutBinary>::Output::HALF
            );
            let length = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(
                (length_i64 as usize).into_scalar()
            );
            let len_sub_one = length._sub(one);
            let two_pi = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(
                <T as FloatOutBinary>::Output::TWOPI
            );
            if per_thread_real_len > 0 {
                data.as_raw_mut()
                    .par_chunks_exact_mut(per_thread_real_len)
                    .enumerate()
                    .for_each(|(idx, lhs)| {
                        assert_eq!(
                            lhs.len() % <<T as FloatOutBinary>::Output as TypeCommon>::Vec::SIZE,
                            0
                        );
                        let idx = idx * per_thread_real_len;
                        let mut idxes =
                            vec![<T as FloatOutBinary>::Output::ZERO; <<T as FloatOutBinary>::Output as TypeCommon>::Vec::SIZE];
                        lhs.chunks_exact_mut(
                            <<T as FloatOutBinary>::Output as TypeCommon>::Vec::SIZE
                        )
                            .enumerate()
                            .for_each(|(i, res)| {
                                let i =
                                    idx +
                                    i * <<T as FloatOutBinary>::Output as TypeCommon>::Vec::SIZE;
                                idxes
                                    .iter_mut()
                                    .enumerate()
                                    .for_each(|(j, x)| {
                                        *x = (i + j).into_scalar();
                                    });
                                let i = unsafe {
                                    <<T as FloatOutBinary>::Output as TypeCommon>::Vec::from_ptr(
                                        idxes.as_ptr()
                                    )
                                };
                                let val = half._sub(
                                    half._mul(two_pi._mul(i)._div(len_sub_one)._cos())
                                );
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        val.as_ptr(),
                                        res.as_mut_ptr(),
                                        <<T as FloatOutBinary>::Output as TypeCommon>::Vec::SIZE
                                    );
                                }
                            });
                    });
            }
            let half = T::HALF;
            let len_sub_one = (length_i64 - 1).into_scalar();
            let two_pi = <T as FloatOutBinary>::Output::TWOPI;
            data.as_raw_mut()
                [data.size() - remain..].iter_mut()
                .enumerate()
                .for_each(|(i, x)| {
                    let i: <T as FloatOutBinary>::Output = i.into_scalar();
                    *x = half._sub(half._mul(two_pi._mul(i)._div(len_sub_one)._cos()));
                });
        }
        #[cfg(not(feature = "simd"))]
        {
            let len_sub_one: <T as FloatOutBinary>::Output = (length_i64 - 1).into_scalar();
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let i: <T as FloatOutBinary>::Output = i.into_scalar();
                    let mul = <T as FloatOutBinary>::Output::TWOPI._mul(i);
                    let div = mul._div(len_sub_one);
                    let cos = div._cos();
                    *x = <T as FloatOutBinary>::Output::HALF._mul(
                        <T as FloatOutBinary>::Output::ONE._sub(cos)
                    );
                });
        }
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
            FloatOutUnary +
            FloatOutBinary +
            FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: FloatOutUnary<Output = <T as FloatOutBinary>::Output> +
            NormalOut<Output = <T as FloatOutBinary>::Output> +
            CommonBounds +
            FloatConst +
            FloatOutBinary<Output = <T as FloatOutBinary>::Output>,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec> +
            FloatOutBinary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec> +
            FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>,
        usize: IntoScalar<T> + IntoScalar<<T as FloatOutBinary>::Output>,
        i64: IntoScalar<T> + IntoScalar<<T as FloatOutBinary>::Output>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hann_window(
        window_length: i64,
        periodic: bool
    ) -> anyhow::Result<Tensor<<T as FloatOutBinary>::Output>> {
        Ok(Tensor::from(_Tensor::hann_window(window_length, periodic)?.into()))
    }
}
