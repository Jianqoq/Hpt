use crate::backends::cpu::kernels::conv2d::utils::handle_post;
use crate::tensor_base::_Tensor;
use hpt_types::REGNUM;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::type_promote::NormalOut;
use hpt_types::vectors::traits::*;
use rayon::prelude::*;

#[track_caller]
pub(crate) fn dwconv2d<T: CommonBounds, const DEVICE: usize, A>(
    img: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    kernel: &_Tensor<T, Cpu, DEVICE, A>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    ShapeError::check_contiguous(
        "pooling input must be contiguous".to_string(),
        &img.layout(),
    )?;
    let img_shape = img.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let in_channels = img_shape[3];
    let kernel_height = kernel.shape()[0];
    let kernel_width = kernel.shape()[1];
    let k_in_channels = kernel.shape()[2];
    let out_channels = kernel.shape()[3];
    if 1 != k_in_channels {
        panic!("kernel in_channel must equal to 1, got {}", k_in_channels);
    }
    if out_channels != in_channels {
        return Err(ShapeError::ConvError {
            message: format!(
                "kernel out_channel {} not match input in_channel {}",
                out_channels, in_channels
            ),
            location: core::panic::Location::caller(),
        }
        .into());
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = img.clone();
    if out_height <= 0 || out_width <= 0 {
        return Err((ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let mut output =
        _Tensor::<T, Cpu, DEVICE, A>::empty([batch, out_height, out_width, in_channels])?;
    let out = output.ptr::<T>();
    let inp = img.ptr();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = img.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let ks0 = kernel.strides()[0]; // kernel_height
    let ks1 = kernel.strides()[1]; // kernel_width

    let kernel = kernel.ptr::<T>();

    let out_size = batch * out_height * out_width;

    const IC_BLOCK_SIZE: usize = REGNUM / 2;
    let in_channel_remain = in_channels % ((IC_BLOCK_SIZE * T::Vec::SIZE) as i64);
    (0..out_size).into_par_iter().for_each(|idx| {
        let out = out.clone();
        let b = idx / (out_height * out_width);
        let h = (idx / out_width) % out_height;
        let w = idx % out_width;

        for ii in (0..in_channels - in_channel_remain).step_by(IC_BLOCK_SIZE * T::Vec::SIZE) {
            let mut res_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
            for kh in 0..kernel_height {
                if h * step_height + kh * dh < ph_start
                    || h * step_height + kh * dh - ph_start >= img_height
                {
                    continue;
                }
                for kw in 0..kernel_width {
                    if w * step_width + kw * dw < pw_start
                        || w * step_width + kw * dw - pw_start >= img_width
                    {
                        continue;
                    }
                    let mut inp_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
                    let mut kernel_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
                    for (idx, (vec, kvec)) in
                        inp_vecs.iter_mut().zip(kernel_vecs.iter_mut()).enumerate()
                    {
                        let i = ii + ((idx * T::Vec::SIZE) as i64);
                        let inp_idx = b * isb
                            + (h * step_height + kh * dh - ph_start) * ish
                            + (w * step_width + kw * dw - pw_start) * isw
                            + i;
                        let kernel_idx = kh * ks0 + kw * ks1 + i;
                        *vec = unsafe { T::Vec::from_ptr(&inp[inp_idx]) };
                        *kvec = unsafe { T::Vec::from_ptr(&kernel[kernel_idx]) };
                    }
                    for idx in 0..IC_BLOCK_SIZE {
                        res_vecs[idx] = inp_vecs[idx]._mul_add(kernel_vecs[idx], res_vecs[idx]);
                    }
                }
            }
            for (idx, vec) in res_vecs.into_iter().enumerate() {
                let i = ii + ((idx * T::Vec::SIZE) as i64);
                let out_idx = b * osb + h * osh + w * osw + i;
                let out_vec = (unsafe { out.ptr.add(out_idx as usize) }) as *mut T::Vec;
                unsafe {
                    out_vec.write_unaligned(vec.read_unaligned());
                }
            }
        }

        let remain = in_channel_remain % (T::Vec::SIZE as i64);
        for ii in (in_channels - in_channel_remain..in_channels - remain).step_by(T::Vec::SIZE) {
            let mut res_vecs = T::Vec::splat(T::ZERO);
            for kh in 0..kernel_height {
                if h * step_height + kh * dh < ph_start
                    || h * step_height + kh * dh - ph_start >= img_height
                {
                    continue;
                }
                for kw in 0..kernel_width {
                    if w * step_width + kw * dw < pw_start
                        || w * step_width + kw * dw - pw_start >= img_width
                    {
                        continue;
                    }
                    let i = ii;
                    let inp_idx = b * isb
                        + (h * step_height + kh * dh - ph_start) * ish
                        + (w * step_width + kw * dw - pw_start) * isw
                        + i;
                    let inp_vec = unsafe { T::Vec::from_ptr(&inp[inp_idx]) };
                    let kernel_idx = kh * ks0 + kw * ks1 + i;
                    let kernel_vec = unsafe { T::Vec::from_ptr(&kernel[kernel_idx]) };
                    res_vecs = inp_vec._mul_add(kernel_vec, res_vecs);
                }
            }
            let i = ii;
            let out_idx = b * osb + h * osh + w * osw + i;
            let out_vec = (unsafe { out.ptr.add(out_idx as usize) }) as *mut T::Vec;
            unsafe {
                out_vec.write_unaligned(res_vecs.read_unaligned());
            }
        }

        for ii in in_channels - remain..in_channels {
            let mut res = T::ZERO;
            for kh in 0..kernel_height {
                if h * step_height + kh * dh < ph_start
                    || h * step_height + kh * dh - ph_start >= img_height
                {
                    continue;
                }
                for kw in 0..kernel_width {
                    if w * step_width + kw * dw < pw_start
                        || w * step_width + kw * dw - pw_start >= img_width
                    {
                        continue;
                    }
                    let i = ii;
                    let inp_idx = b * isb
                        + (h * step_height + kh * dh - ph_start) * ish
                        + (w * step_width + kw * dw - pw_start) * isw
                        + i;
                    let inp_val = inp[inp_idx];
                    let kernel_idx = kh * ks0 + kw * ks1 + i;
                    let kernel_val = kernel[kernel_idx];
                    res = inp_val._mul_add(kernel_val, res);
                }
            }
            let i = ii;
            let out_idx = b * osb + h * osh + w * osw + i;
            let out_ptr = (unsafe { out.ptr.add(out_idx as usize) }) as *mut T;
            unsafe {
                out_ptr.write_unaligned(res);
            }
        }
    });

    handle_post(&mut output, bias, post_scalar, post_vec)?;

    Ok(output)
}
