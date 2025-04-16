use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::calculate_kernel_params;
use super::utils::create_packed_kernel;
use super::utils::handle_post;
use super::utils::pack_kernel_mp;
use crate::backends::common::conv::cal_conv2d_output_shape;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOutPromote;
use hpt_types::vectors::traits::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

type IM<T> = <T as NormalOutPromote>::Intermediate;
type IMVec<T> = <<T as NormalOutPromote>::Intermediate as TypeCommon>::Vec;

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    vec_cast_back: fn(*const IMVec<T>) -> T::Vec,
    vec_cast: fn(*const T) -> IMVec<T>,
    cast: fn(T) -> IM<T>,
    cast_back: fn(IM<T>) -> T,
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    bool: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
    T: Cast<IM<T>>,
    IM<T>: CommonBounds + Cast<T>,
{
    ShapeError::check_contiguous(
        "Conv2d requires input tensor to be contiguous. ".to_string(),
        input.layout(),
    )?;
    ShapeError::check_contiguous(
        "Conv2d requires kernel tensor to be contiguous. ".to_string(),
        kernels.layout(),
    )?;
    if bias.is_some() {
        ShapeError::check_contiguous(
            "Conv2d requires bias tensor to be contiguous. ".to_string(),
            bias.unwrap().layout(),
        )?;
    }
    let img_shape = input.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
    let kernel_shape = kernels.shape();
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        return Err((ShapeError::ConvError {
            message: format!(
                "kernel in_channel {} not match input in_channel {}",
                in_channels, img_channels
            ),
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let (out_height, out_width) = cal_conv2d_output_shape(
        img_height,
        img_width,
        kh,
        kw,
        &[(ph_start, ph_end), (pw_start, pw_end)],
        &[step_height, step_width],
        &[dh, dw],
    );

    let casted_input = _Tensor::<IM<T>, Cpu, DEVICE, A>::empty(input.shape())?;
    let buffer_slice =
        unsafe { std::slice::from_raw_parts(input.ptr().ptr as *mut T, input.size()) };
    let out_slice = unsafe {
        std::slice::from_raw_parts_mut(casted_input.ptr().ptr as *mut IM<T>, input.size())
    };
    let mut chunk_exact = out_slice.par_chunks_exact_mut(T::Vec::SIZE);
    let chunk_buffer_exact = buffer_slice.par_chunks_exact(T::Vec::SIZE);
    chunk_exact
        .remainder()
        .into_par_iter()
        .zip(chunk_buffer_exact.remainder().into_par_iter())
        .for_each(|(out, buffer)| {
            *out = cast(*buffer);
        });

    match IM::<T>::BYTE_SIZE / T::BYTE_SIZE {
        2 => {
            chunk_exact
                .into_par_iter()
                .zip(chunk_buffer_exact.into_par_iter())
                .for_each(|(out, buffer)| {
                    let out_ptr = out.as_mut_ptr() as *mut IMVec<T>;
                    let buffer_ptr = buffer.as_ptr() as *const T;
                    unsafe {
                        seq_macro::seq!(N in 0..2 {
                            let buffer_ptr = buffer_ptr.add(N * IMVec::<T>::SIZE);
                            out_ptr.add(N).write_unaligned(vec_cast(buffer_ptr));
                        });
                    }
                });
        }
        4 => {
            chunk_exact
                .into_par_iter()
                .zip(chunk_buffer_exact.into_par_iter())
                .for_each(|(out, buffer)| {
                    let out_ptr = out.as_ptr() as *mut IMVec<T>;
                    let buffer_ptr = buffer.as_ptr() as *const T;
                    unsafe {
                        seq_macro::seq!(N in 0..4 {
                            let buffer_ptr = buffer_ptr.add(N * IMVec::<T>::SIZE);
                            out_ptr.add(N).write_unaligned(vec_cast(buffer_ptr));
                        });
                    }
                });
        }
        _ => {
            unreachable!()
        }
    }

    let img = casted_input.clone();
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
    let mut output = _Tensor::<T, Cpu, DEVICE, A>::empty([batch, out_height, out_width, out_channels])?;
    let out = output.ptr();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = img.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels

    let outer = batch * out_height;

    let inp_ptr = img.ptr();
    let kernel_ptr = kernels.ptr();
    let nr = T::get_max_mixed_precision_nr() * T::Vec::SIZE;
    let mr = T::get_max_mixed_precision_mr().min(out_width as usize);
    let param = calculate_kernel_params::<T>(
        in_channels,
        out_channels,
        out_width,
        mr,
        nr,
        [kh as usize, kw as usize],
    );

    let kc: i64 = param.mc as i64;
    let ic: i64 = param.kc as i64;
    let oc: i64 = param.nc as i64;
    let buffer =
        create_packed_kernel::<IM<T>, DEVICE, A>(kh, kw, in_channels, out_channels, oc, nr as i64)?;
    pack_kernel_mp(
        buffer.ptr(),
        kernel_ptr,
        in_channels,
        out_channels,
        ic,
        oc,
        nr as i64,
        [kh, kw],
        [ks0, ks1, ks2],
    );

    let get_kernel = if ph_start == 0 && pw_start == 0 && ph_end == 0 && pw_end == 0 {
        T::get_mixed_precision_kernel
    } else {
        T::get_mixed_precision_kernel_with_padding
    };

    (0..outer).into_par_iter().for_each(|idx| {
        let kernel = buffer.ptr();
        let b = idx / out_height;
        let ll = idx % out_height;

        let inp = inp_ptr.clone() + b * isb;
        let out = out.clone() + b * osb + ll * osh;

        for k in (0..out_width).step_by(kc as usize) {
            let owb = kc.min(out_width - k);
            let mut kernel_idx: i64 = 0;
            for i in (0..in_channels).step_by(ic as usize) {
                let icb = ic.min(in_channels - i);
                for j in (0..out_channels).step_by(oc as usize) {
                    let ocb = oc.min(out_channels - j);

                    let kernel_idx_1 = kernel_idx;
                    for kk in (0..owb).step_by(mr as usize) {
                        let owr = (mr as i64).min(owb - kk);
                        let micro_kernel = get_kernel(nr / <T>::Vec::SIZE, owr as usize);
                        kernel_idx = kernel_idx_1;
                        for jj in (0..ocb).step_by(nr as usize) {
                            let ocr = (nr as i64).min(ocb - jj);
                            micro_kernel(
                                inp + i,
                                kernel,
                                out,
                                icb,
                                osw,
                                &mut kernel_idx,
                                [kk + k, jj + j, ll],
                                [kh, kw],
                                [step_height, step_width],
                                [ph_start, pw_start],
                                [img_height, img_width],
                                [ish, isw],
                                [owr, ocr],
                                [dh, dw],
                                i == 0,
                                vec_cast,
                                vec_cast_back,
                                cast,
                                cast_back,
                            );
                        }
                    }
                }
            }
        }
    });

    handle_post(&mut output, bias, post_scalar, post_vec)?;

    Ok(output)
}
