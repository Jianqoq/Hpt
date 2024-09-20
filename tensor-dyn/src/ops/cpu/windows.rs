use std::ops::{Mul, Sub};

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use tensor_traits::{CommonBounds, TensorCreator, TensorLike};
use tensor_types::{
    dtype::{FloatConst, TypeCommon},
    into_scalar::IntoScalar,
    traits::{Init, VecCommon},
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};

use crate::{tensor::Tensor, tensor_base::_Tensor};

type Simd<T> = <<T as FloatOutBinary>::Output as TypeCommon>::Vec;
type FBO<T> = <T as FloatOutBinary>::Output;

impl<T> _Tensor<T>
where
    f64: IntoScalar<FBO<T>>,
    T: CommonBounds + FloatOutBinary,
    FBO<T>: CommonBounds
        + FloatOutUnary<Output = FBO<T>>
        + Mul<Output = FBO<T>>
        + Sub<Output = FBO<T>>
        + FloatConst,
    FBO<T>: std::ops::Neg<Output = FBO<T>>,
    FBO<T>: NormalOut<FBO<T>, Output = FBO<T>> + FloatOutBinary<FBO<T>, Output = FBO<T>>,
    Simd<T>: NormalOut<Simd<T>, Output = Simd<T>>
        + FloatOutBinary<Simd<T>, Output = Simd<T>>
        + FloatOutUnary<Output = Simd<T>>,
    usize: IntoScalar<FBO<T>>,
    i64: IntoScalar<T>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hamming_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<FBO<T>>> {
        Self::__hamming_window(
            window_length,
            0.54.into_scalar(),
            0.46.into_scalar(),
            periodic,
        )
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hann_window(window_length: i64, periodic: bool) -> anyhow::Result<_Tensor<FBO<T>>> {
        Self::__hamming_window(
            window_length,
            0.5.into_scalar(),
            0.5.into_scalar(),
            periodic,
        )
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn __hamming_window(
        window_length: i64,
        alpha: FBO<T>,
        beta: FBO<T>,
        periodic: bool,
    ) -> anyhow::Result<_Tensor<FBO<T>>> {
        let length_usize = (if periodic {
            window_length
        } else {
            window_length - 1
        }) as usize;
        let length: FBO<T> = length_usize.into_scalar();
        let mut ret = _Tensor::<FBO<T>>::empty(&[length_usize as i64])?;
        let mut chunk_exact = ret
            .as_raw_mut()
            .par_chunks_exact_mut(<Simd<T> as VecCommon>::SIZE);
        let two_pi = Simd::<T>::splat(FBO::<T>::TWOPI);
        let length_vec = Simd::<T>::splat(length);
        let alpha_vec = Simd::<T>::splat(alpha);
        let beta_vec = Simd::<T>::splat(-beta);
        let remainder = chunk_exact.remainder();
        remainder.iter_mut().enumerate().for_each(|(idx, x)| {
            let idx: FBO<T> = idx.into_scalar();
            *x = idx
                ._mul(FBO::<T>::TWOPI._div(length))
                ._cos()
                ._mul_add(-beta, alpha);
        });
        chunk_exact.enumerate().for_each(|(x, vec)| {
            let idx = x * <Simd<T> as VecCommon>::SIZE;
            let mut idxes = Simd::<T>::splat(FBO::<T>::ZERO);
            for i in 0..<Simd<T> as VecCommon>::SIZE {
                idxes[i] = (idx + i).into_scalar();
            }
            let ptr = vec as *mut _ as *mut Simd<T>;

            let res = tensor_types::traits::VecTrait::_mul_add(
                idxes._mul(two_pi._div(length_vec))._cos(),
                beta_vec,
                alpha_vec,
            );
            unsafe {
                ptr.write(res);
            }
        });
        Ok(ret)
    }
}

impl<T> Tensor<T>
where
    f64: IntoScalar<FBO<T>>,
    T: CommonBounds + FloatOutBinary,
    FBO<T>: CommonBounds
        + FloatOutUnary<Output = FBO<T>>
        + Mul<Output = FBO<T>>
        + Sub<Output = FBO<T>>
        + FloatConst,
    FBO<T>: std::ops::Neg<Output = FBO<T>>,
    FBO<T>: NormalOut<FBO<T>, Output = FBO<T>> + FloatOutBinary<FBO<T>, Output = FBO<T>>,
    Simd<T>: NormalOut<Simd<T>, Output = Simd<T>>
        + FloatOutBinary<Simd<T>, Output = Simd<T>>
        + FloatOutUnary<Output = Simd<T>>,
    usize: IntoScalar<FBO<T>>,
    i64: IntoScalar<T>,
{
    /// Generates a Hamming window of a specified length.
    ///
    /// The `hamming_window` function creates a Hamming window, which is commonly used in signal processing for smoothing data or reducing spectral leakage.
    ///
    /// The Hamming window is mathematically defined as:
    ///
    /// ```text
    /// w(n) = 0.54 - 0.46 * cos(2πn / (N - 1))
    /// ```
    ///
    /// where `N` is the `window_length` and `n` ranges from 0 to `N-1`.
    ///
    /// # Parameters
    ///
    /// - `window_length`: The length of the window.
    /// - `periodic`: If `true`, creates a periodic window, suitable for use in spectral analysis.
    ///
    /// # Returns
    ///
    /// - A tensor containing the Hamming window.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hamming_window(window_length: i64, periodic: bool) -> anyhow::Result<Tensor<FBO<T>>> {
        Ok(Tensor::from(
            _Tensor::hamming_window(window_length, periodic)?.into(),
        ))
    }

    /// Generates a Hann window of a specified length.
    ///
    /// The `hann_window` function creates a Hann window, which is used in signal processing to taper data and reduce spectral leakage.
    ///
    /// The Hann window is mathematically defined as:
    ///
    /// ```text
    /// w(n) = 0.5 * (1 - cos(2πn / (N - 1)))
    /// ```
    ///
    /// where `N` is the `window_length` and `n` ranges from 0 to `N-1`.
    ///
    /// # Parameters
    ///
    /// - `window_length`: The length of the window.
    /// - `periodic`: If `true`, creates a periodic window, suitable for use in spectral analysis.
    ///
    /// # Returns
    ///
    /// - A tensor containing the Hann window.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hann_window(window_length: i64, periodic: bool) -> anyhow::Result<Tensor<FBO<T>>> {
        Ok(Tensor::from(
            _Tensor::hann_window(window_length, periodic)?.into(),
        ))
    }
}
