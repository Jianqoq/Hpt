use crate::{ops::cpu::tensor_internal::windows::Simd, tensor::Tensor, tensor_base::_Tensor, Cpu};
use std::ops::{Mul, Sub};
use tensor_common::error::base::TensorError;
use tensor_traits::{CommonBounds, WindowOps};
use tensor_types::{
    dtype::FloatConst,
    into_scalar::IntoScalar,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};

type FBO<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize> WindowOps for Tensor<T, Cpu, DEVICE>
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
    type Output = Tensor<FBO<T>, Cpu, DEVICE>;
    type Meta = T;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hamming_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(Tensor::from(
            _Tensor::hamming_window(window_length, periodic)?.into(),
        ))
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hann_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(Tensor::from(
            _Tensor::hann_window(window_length, periodic)?.into(),
        ))
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn blackman_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError>
    where
        T: FloatConst,
        i64: IntoScalar<<T as FloatOutBinary>::Output>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::blackman_window(window_length, periodic)?.into())
    }
}
