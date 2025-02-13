use crate::{ops::cpu::tensor_internal::windows::Simd, tensor::Tensor, tensor_base::_Tensor, Cpu};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, WindowOps};
use hpt_types::{
    dtype::FloatConst,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};
use std::ops::{Mul, Sub};

type FBO<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize> WindowOps for Tensor<T, Cpu, DEVICE>
where
    f64: Cast<FBO<T>>,
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
    usize: Cast<FBO<T>>,
    i64: Cast<T>,
{
    type Output = Tensor<FBO<T>, Cpu, DEVICE>;
    type Meta = T;
    #[track_caller]
    fn hamming_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(Tensor::from(
            _Tensor::hamming_window(window_length, periodic)?.into(),
        ))
    }
    #[track_caller]
    fn hann_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError> {
        Ok(Tensor::from(
            _Tensor::hann_window(window_length, periodic)?.into(),
        ))
    }
    #[track_caller]
    fn blackman_window(window_length: i64, periodic: bool) -> Result<Self::Output, TensorError>
    where
        T: FloatConst,
        i64: Cast<<T as FloatOutBinary>::Output>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::blackman_window(window_length, periodic)?.into())
    }
}
