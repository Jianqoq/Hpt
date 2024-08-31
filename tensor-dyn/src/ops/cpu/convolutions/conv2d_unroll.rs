use crate::tensor_base::_Tensor;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::vectors::traits::*;

#[cfg(target_feature = "fma")]
pub fn conv2d_ex<
    T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign,
    const REGNUM: usize,
    const VECSIZE: usize,
    VEC
    >(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where VEC: VecTrait<T> + Copy + Init<T>, T: IntoScalar<T>
{
    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
        img.pad(
            &[
                (ph_start, ph_end),
                (pw_start, pw_end),
                (0, 0),
            ],
            T::ZERO
        )?
    } else {
        img.clone()
    };
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    for l in 0..out_height {
        for n in 0..kernel_height {
            for m in 0..kernel_width {
                for i in 0..in_channels {
                    for k in 0..out_width {
                        for j in 0..out_channels {
                            let k_val = kernel[n * ks0 + m * ks1 + i * ks2 + j * ks3];
                            let i_val = inp[(l * step_height + n) * is0 + (k * step_width + m) * is1 + i * is2]; // prettier-ignore
                            out[l * os0 + k * os1 + j * os2] += i_val * k_val; // prettier-ignore
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}
