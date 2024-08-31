use std::ops::AddAssign;
use std::ops::Mul;

use crate::tensor_base::_Tensor;
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

#[cfg(target_feature = "fma")]
pub fn conv2d_intel<
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
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
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
    let output = _Tensor::<T>::zeros([batch, out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let obs = output.strides()[0]; // batch
    let ohs = output.strides()[1]; // height
    let ows = output.strides()[2]; // width
    let ocs = output.strides()[3]; // channels

    let ibs = img.strides()[0]; // batch
    let ihs = img.strides()[1]; // height
    let iws = img.strides()[2]; // width
    let ics = img.strides()[3]; // channels

    let khs = kernels.strides()[0]; // kernel_height
    let kws = kernels.strides()[1]; // kernel_width
    let kis = kernels.strides()[2]; // in_channels
    let kos = kernels.strides()[3]; // out_channels

    for b in 0..batch {
        for o in 0..out_channels {
            for i in 0..in_channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let ij = h * step_height;
                        let jw = w * step_width;
                        for k in 0..kernel_height {
                            for l in 0..kernel_width {
                                let i_val =
                                    inp[b * ibs + i * ics + (ij + k) * ihs + (jw + l) * iws];
                                let k_val = kernel[o * kos + i * kis + k * khs + l * kws];
                                out[b * obs + o * ocs + h * ohs + w * ows] += i_val * k_val;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

#[cfg(target_feature = "fma")]
pub fn conv2d_intel_reg_blocking<
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
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
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
    let output = _Tensor::<T>::zeros([batch, out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let obs = output.strides()[0]; // batch
    let ohs = output.strides()[1]; // height
    let ows = output.strides()[2]; // width
    let ocs = output.strides()[3]; // channels

    let ibs = img.strides()[0]; // batch
    let ihs = img.strides()[1]; // height
    let iws = img.strides()[2]; // width
    let ics = img.strides()[3]; // channels

    let khs = kernels.strides()[0]; // kernel_height
    let kws = kernels.strides()[1]; // kernel_width
    let kis = kernels.strides()[2]; // in_channels
    let kos = kernels.strides()[3]; // out_channels

    let rbp = 2;
    let rbq = 2;
    let cb = in_channels / (VECSIZE as i64);
    let ob = out_channels / (VECSIZE as i64);
    let ohb = out_height / rbp;
    let owb = out_width / rbq;

    for b in 0..batch {
        for o in 0..ob {
            for i in 0..cb {
                for h in 0..ohb {
                    for w in 0..owb {
                        let ij = h * step_height * rbp;
                        let ii = w * step_width * rbq;
                        let oj = h * rbp;
                        let oi = w * rbq;
                        let kernel = kernel.ptr.wrapping_offset(
                            (o * (VECSIZE as i64) * kos + i * (VECSIZE as i64) * kis) as isize
                        );
                        let inp = inp.ptr.wrapping_offset(
                            (b * ibs + i * (VECSIZE as i64) * ics) as isize
                        );
                        let out = out.ptr.wrapping_offset(
                            (b * obs + o * (VECSIZE as i64) * ocs + oj * ohs + oi * ows) as isize
                        );
                        conv_kernel::<T, VECSIZE>(
                            [khs, kws, kis, kos],
                            [ihs, iws, ics],
                            [ohs, ows, ocs],
                            [kernel, inp, out],
                            [ij, ii],
                            [kernel_height, kernel_width, step_height, step_width],
                            [rbp, rbq]
                        );
                    }
                }
            }
        }
    }

    Ok(output)
}

#[inline(always)]
pub fn conv_kernel<T, const VECSIZE: usize>(
    [khs, kws, kis, kos]: [i64; 4],
    [ihs, iws, ics]: [i64; 3],
    [ohs, ows, ocs]: [i64; 3],
    [kernel, inp, out]: [*const T; 3],
    [ij, ii]: [i64; 2],
    [kernel_height, kernel_width, step_height, step_width]: [i64; 4],
    [rbp, rbq]: [i64; 2]
)
    where T: Mul<Output = T> + AddAssign + Copy
{
    let out = out as *mut T;
    for k in 0..kernel_height {
        for l in 0..kernel_width {
            for kk in 0..VECSIZE as i64 {
                for cc in 0..VECSIZE as i64 {
                    unsafe {
                        let k_val = *kernel.wrapping_offset((kk * kos + cc * kis + k * khs + l * kws) as isize); // prettier-ignore
                        for p in 0..rbp {
                            for q in 0..rbq {
                                let ij_p = ij + step_height * p;
                                let ii_p = ii + step_width * q;
                                let i_val = *inp.wrapping_offset((cc * ics + (ij_p + k) * ihs + (ii_p + l) * iws) as isize); // prettier-ignore
                                *out.wrapping_offset((kk * ocs + p * ohs + q * ows) as isize) += i_val * k_val; // prettier-ignore
                            }
                        }
                    }
                }
            }
        }
    }
}
