use crate::backends::common::conv::cal_conv2d_output_shape;
use crate::backends::cpu::cache_utils::cache::Cache;
use crate::backends::cpu::kernels::conv2d::micro_kernels::conv::Params;
use crate::backends::cpu::kernels::conv2d::micro_kernels::conv::PartialParams;
use crate::tensor_base::_Tensor;
use crate::ALIGN;
use crate::REGNUM;
use gemm_common::cache::DivCeil;
use gemm_common::cache::KernelParams;
use gemm_common::cache::CACHE_INFO;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::utils::pointer::Pointer;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::vectors::traits::*;
use rayon::prelude::*;

use super::microkernel_trait::Conv2dMicroKernel;

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    activation: Option<fn(T::Vec) -> T::Vec>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    bool: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
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
    let img = input.clone();
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
    let output = _Tensor::<T, Cpu, DEVICE, A>::zeros([batch, out_height, out_width, out_channels])?;
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

    let inp_ptr = input.ptr();
    let kernel_ptr = kernels.ptr();
    let nr = (T::get_max_nr()) * T::Vec::SIZE;
    let mr = T::get_max_mr().min(out_width as usize);
    let mut param = if in_channels <= 64 && out_channels <= 64 {
        // skip expensive kernel_params call for small sizes
        let kc = in_channels.min(512);
        let alloc = CACHE_INFO[1].cache_bytes / core::mem::size_of::<T>();
        let nc = (alloc / (kc as usize) / (nr as usize)) * (nr as usize);
        KernelParams {
            kc: kc as usize,
            mc: (out_width as usize).msrv_next_multiple_of(mr as usize),
            nc,
        }
    } else {
        gemm_common::cache::kernel_params(
            out_channels as usize,
            out_width as usize,
            in_channels as usize,
            nr as usize,
            mr as usize,
            std::mem::size_of::<T>(),
        )
    };
    if param.nc == 0 {
        param.nc = (out_channels as usize).msrv_next_multiple_of(nr as usize);
    }
    if param.mc == 0 {
        param.mc = (out_width as usize).msrv_next_multiple_of(mr as usize);
    }

    let kc: i64 = param.nc as i64;
    let ic: i64 = param.kc as i64;
    let oc: i64 = param.mc as i64;
    // println!("kc: {}, ic: {}, nr: {}, oc: {}", kc, ic, nr, oc);
    let packed_kernel_size = kh
        * kw
        * in_channels
        * (out_channels as usize).div_ceil(oc as usize) as i64
        * (oc as usize).div_ceil(nr as usize) as i64
        * nr as i64
        * (std::mem::size_of::<T>() as i64);

    let packed_kernel_raw = unsafe {
        std::alloc::alloc(
            std::alloc::Layout::from_size_align(packed_kernel_size as usize, ALIGN).unwrap(),
        )
    };
    // println!("kc: {}, ic: {}, oc: {}", kc, ic, oc);

    fn pack_kernel<T: CommonBounds>(
        mut packed_kernel: Pointer<T>,
        kernel: Pointer<T>,
        in_channels: i64,
        out_channels: i64,
        ic: i64,
        oc: i64,
        or: i64,
        [kh, kw]: [i64; 2],
        [ks0, ks1, ks2]: [i64; 3],
    ) {
        let mut idx: i64 = 0;
        for i in (0..in_channels).step_by(ic as usize) {
            let icb = ic.min(in_channels - i);
            for j in (0..out_channels).step_by(oc as usize) {
                let ocb = oc.min(out_channels - j);
                for jj in (0..ocb).step_by(or as usize) {
                    let ocr = or.min(ocb - jj);
                    for n in 0..kh {
                        for m in 0..kw {
                            for ii in 0..icb {
                                for nr in 0..ocr {
                                    packed_kernel[idx] =
                                        kernel[n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr];
                                    idx += 1;
                                }
                                for _ in ocr..or {
                                    packed_kernel[idx] = T::ZERO;
                                    idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "bound_check")]
    let packed_kernel = Pointer::new(
        packed_kernel_raw as *mut T,
        packed_kernel_size / std::mem::size_of::<T>() as i64,
    );
    #[cfg(not(feature = "bound_check"))]
    let packed_kernel = Pointer::new(packed_kernel_raw as *mut T);
    pack_kernel(
        packed_kernel,
        kernel_ptr,
        in_channels,
        out_channels,
        ic,
        oc,
        nr as i64,
        [kh, kw],
        [ks0, ks1, ks2],
    );

    (0..outer).into_par_iter().for_each(|idx| {
        let kernel = packed_kernel;
        let b = idx / out_height;
        let ll = idx % out_height;

        let inp = inp_ptr.clone() + b * isb + ll * step_height * ish;
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
                        let micro_kernel = T::get_kernel(nr / <T>::Vec::SIZE, owr as usize);
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
                                i == 0,
                            );
                        }
                    }
                }
            }
        }
    });

    unsafe {
        std::alloc::dealloc(
            packed_kernel_raw as *mut _,
            std::alloc::Layout::from_size_align(packed_kernel_size as usize, ALIGN).unwrap(),
        );
    }
    Ok(output)
}
