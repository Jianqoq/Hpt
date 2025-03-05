use crate::ops::cuda::{cuda_utils::get_module_name_1, utils::unary::unary::uary_fn_with_out_simd};
use crate::{tensor_base::_Tensor, Cuda};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, TensorCreator, WindowOps};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::{
    dtype::FloatConst,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary},
};

impl<T, const DEVICE_ID: usize, Al> WindowOps for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    f64: Cast<T>,
    T: CommonBounds
        + FloatOutBinary<Output = T>
        + FloatOutUnary<Output = T>
        + FloatConst
        + DeviceRepr
        + CudaType,
    usize: Cast<T>,
    i64: Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cuda, DEVICE_ID, Al>;
    type Meta = T;
    #[track_caller]
    fn hamming_window(
        window_length: i64,
        periodic: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        __hamming_window::<T, DEVICE_ID, Al>(window_length, (0.54).cast(), (0.46).cast(), periodic)
    }

    #[track_caller]
    fn hann_window(
        window_length: i64,
        periodic: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        __hamming_window::<T, DEVICE_ID, Al>(window_length, (0.5).cast(), (0.5).cast(), periodic)
    }

    #[track_caller]
    fn blackman_window(
        window_length: i64,
        periodic: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let length_usize = if periodic {
            window_length
        } else {
            window_length - 1
        };
        let length: T = length_usize.cast();
        let ret = _Tensor::<T, Cuda, DEVICE_ID, Al>::empty(&[length_usize])?;
        uary_fn_with_out_simd(
            &ret,
            &get_module_name_1("blackman_window", &ret),
            |out, idx| {
                let res = match T::STR {
                    "f32" => {
                        format!(
                            "
                            float n = (float){idx};
                            float w1 = 2.0f * M_PI * n / {length}f;
                            float w2 = 2.0f * w1;  // 4π * n / (N-1)
                            {out} = 0.42f - 0.5f * cosf(w1) + 0.08f * cosf(w2);"
                        )
                    }
                    "f64" => {
                        format!(
                            "
                            double n = (double){idx};
                            double w1 = 2.0 * M_PI * n / {length};
                            double w2 = 2.0 * w1;
                            {out} = 0.42 - 0.5 * cos(w1) + 0.08 * cos(w2);"
                        )
                    }
                    "f16" => {
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
                Scalar::<T>::new(res)
            },
            None::<Self::Output>,
        )
    }
}

#[track_caller]
fn __hamming_window<T, const DEVICE_ID: usize, Al>(
    window_length: i64,
    alpha: T,
    beta: T,
    periodic: bool,
) -> std::result::Result<_Tensor<T, Cuda, DEVICE_ID, Al>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType,
    usize: Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let length_usize = (if periodic {
        window_length
    } else {
        window_length - 1
    }) as usize;
    let length: T = length_usize.cast();
    let ret = _Tensor::<T, Cuda, DEVICE_ID, Al>::empty(&[length_usize as i64])?;
    uary_fn_with_out_simd(
        &ret,
        &get_module_name_1("hamming_window", &ret),
        |out, idx| {
            let res = match T::STR {
                "f32" => {
                    format!(
                        "
                    float n = (float){idx};
                    {out} = {alpha} - {beta} * cosf(2.0f * M_PI * n / {length});"
                    )
                }
                "f64" => {
                    format!(
                        "
                    double n = (double){idx};
                    {out} = {alpha} - {beta} * cos(2.0 * M_PI * n / {length});"
                    )
                }
                "f16" => {
                    format!(
                        "
                    float n = (float){idx};
                    {out} = __float2half({alpha}f - {beta}f * cosf(2.0f * M_PI * n / {length}));"
                    )
                }
                _ => unreachable!(),
            };
            Scalar::<T>::new(res)
        },
        None::<_Tensor<T, Cuda, DEVICE_ID, Al>>,
    )
}
