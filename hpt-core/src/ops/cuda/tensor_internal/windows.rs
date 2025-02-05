use crate::ops::cuda::{cuda_utils::get_module_name_1, unary::uary_fn_with_out_simd};
use crate::{tensor_base::_Tensor, Cuda};
use cudarc::driver::DeviceRepr;
use hpt_common::err_handler::TensorError;
use hpt_traits::{CommonBounds, TensorCreator};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::Dtype::*;
use hpt_types::{
    cast::Cast,
    dtype::{FloatConst, TypeCommon},
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};
use std::ops::{Mul, Sub};

pub(crate) type Simd<T> = <<T as FloatOutBinary>::Output as TypeCommon>::Vec;
type FBO<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID>
where
    f64: Cast<FBO<T>>,
    T: CommonBounds + FloatOutBinary + DeviceRepr,
    FBO<T>: CommonBounds
        + FloatOutUnary<Output = FBO<T>>
        + Mul<Output = FBO<T>>
        + Sub<Output = FBO<T>>
        + FloatConst
        + DeviceRepr,
    FBO<T>: std::ops::Neg<Output = FBO<T>>,
    FBO<T>: NormalOut<FBO<T>, Output = FBO<T>> + FloatOutBinary<FBO<T>, Output = FBO<T>>,
    Simd<T>: NormalOut<Simd<T>, Output = Simd<T>>
        + FloatOutBinary<Simd<T>, Output = Simd<T>>
        + FloatOutUnary<Output = Simd<T>>,
    usize: Cast<FBO<T>>,
    i64: Cast<T>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub(crate) fn hamming_window(
        window_length: i64,
        periodic: bool,
    ) -> std::result::Result<_Tensor<FBO<T>, Cuda, DEVICE_ID>, TensorError> {
        Self::__hamming_window(window_length, (0.54).cast(), (0.46).cast(), periodic)
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    pub(crate) fn hann_window(
        window_length: i64,
        periodic: bool,
    ) -> std::result::Result<_Tensor<FBO<T>, Cuda, DEVICE_ID>, TensorError> {
        Self::__hamming_window(window_length, (0.5).cast(), (0.5).cast(), periodic)
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn __hamming_window(
        window_length: i64,
        alpha: FBO<T>,
        beta: FBO<T>,
        periodic: bool,
    ) -> std::result::Result<_Tensor<FBO<T>, Cuda, DEVICE_ID>, TensorError> {
        let length_usize = (if periodic {
            window_length
        } else {
            window_length - 1
        }) as usize;
        let length: FBO<T> = length_usize.cast();
        let ret = _Tensor::<FBO<T>, Cuda, DEVICE_ID>::empty(&[length_usize as i64])?;
        uary_fn_with_out_simd(
            &ret,
            &get_module_name_1("hamming_window", &ret),
            |out, idx| {
                let res = match T::ID {
                    F32 => {
                        format!(
                            "
                        float n = (float){idx};
                        {out} = {alpha} - {beta} * cosf(2.0f * M_PI * n / {length});"
                        )
                    }
                    F64 => {
                        format!(
                            "
                        double n = (double){idx};
                        {out} = {alpha} - {beta} * cos(2.0 * M_PI * n / {length});"
                        )
                    }
                    F16 => {
                        format!("
                        float n = (float){idx};
                        {out} = __float2half({alpha}f - {beta}f * cosf(2.0f * M_PI * n / {length}));")
                    }
                    _ => unreachable!(),
                };
                Scalar::<FBO<T>>::new(res)
            },
            None::<_Tensor<FBO<T>, Cuda, DEVICE_ID>>,
        )
    }

    /// Generates a Blackman window tensor.
    ///
    /// A Blackman window is commonly used in signal processing to reduce spectral leakage.
    /// This method generates a tensor representing the Blackman window, which can be used
    /// for tasks like filtering or analysis in the frequency domain. The window can be
    /// either periodic or symmetric, depending on the `periodic` parameter.
    ///
    /// # Arguments
    ///
    /// * `window_length` - The length of the window, specified as an `i64`. This determines
    ///   the number of elements in the output tensor.
    /// * `periodic` - A boolean flag indicating whether the window should be periodic or symmetric:
    ///   - If `true`, the window will be periodic, which is typically used for spectral analysis.
    ///   - If `false`, the window will be symmetric, which is typically used for filtering.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `<T as FloatOutBinary>::Output`
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn blackman_window(
        window_length: i64,
        periodic: bool,
    ) -> std::result::Result<_Tensor<<T as FloatOutBinary>::Output, Cuda, DEVICE_ID>, TensorError>
    where
        T: FloatConst,
        i64: Cast<<T as FloatOutBinary>::Output>,
    {
        let length_usize = if periodic {
            window_length
        } else {
            window_length - 1
        };
        let length: <T as FloatOutBinary>::Output = length_usize.cast();
        let ret =
            _Tensor::<<T as FloatOutBinary>::Output, Cuda, DEVICE_ID>::empty(&[length_usize])?;
        uary_fn_with_out_simd(
            &ret,
            &get_module_name_1("blackman_window", &ret),
            |out, idx| {
                let res = match T::ID {
                    F32 => {
                        format!(
                            "
                            float n = (float){idx};
                            float w1 = 2.0f * M_PI * n / {length}f;
                            float w2 = 2.0f * w1;  // 4π * n / (N-1)
                            {out} = 0.42f - 0.5f * cosf(w1) + 0.08f * cosf(w2);"
                        )
                    }
                    F64 => {
                        format!(
                            "
                            double n = (double){idx};
                            double w1 = 2.0 * M_PI * n / {length};
                            double w2 = 2.0 * w1;
                            {out} = 0.42 - 0.5 * cos(w1) + 0.08 * cos(w2);"
                        )
                    }
                    F16 => {
                        format!(
                            "
                            float n = (float){idx};
                            float w1 = 2.0f * M_PI * n / {length}f;
                            float w2 = 2.0f * w1;
                            {out} = __float2half(0.42f - 0.5f * cosf(w1) + 0.08f * cosf(w2));"
                        )
                    }
                    _ => unreachable!(),
                };
                Scalar::<FBO<T>>::new(res)
            },
            None::<_Tensor<FBO<T>, Cuda, DEVICE_ID>>,
        )
    }
}
