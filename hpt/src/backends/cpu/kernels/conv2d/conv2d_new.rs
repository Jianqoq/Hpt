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
    let activation = activation.unwrap_or(|x| x);
    let output = _Tensor::<T, Cpu, DEVICE, A>::zeros([batch, out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();

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
    let packed_kernel_size = kh
        * kw
        * in_channels
        * (out_channels as usize).next_multiple_of(nr) as i64
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
                    if ocr == or {
                        for n in 0..kh {
                            for m in 0..kw {
                                for ii in 0..icb {
                                    for nr in 0..ocr {
                                        packed_kernel[idx] = kernel
                                            [n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr];
                                        idx += 1;
                                    }
                                }
                            }
                        }
                    } else {
                        for n in 0..kh {
                            for m in 0..kw {
                                for ii in 0..icb {
                                    for nr in 0..ocr {
                                        packed_kernel[idx] = kernel
                                            [n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr];
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
    }

    #[cfg(feature = "bound_check")]
    let packed_kernel = Pointer::new(packed_kernel_raw as *mut T, packed_kernel_size);
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
        // let packed_a_size = kh * kw * ic * kc * (std::mem::size_of::<T>() as i64);
        // let packed_inp_raw = (unsafe {
        //     std::alloc::alloc(
        //         std::alloc::Layout::from_size_align(packed_a_size as usize, ALIGN).unwrap(),
        //     )
        // }) as *mut T;
        // #[cfg(feature = "bound_check")]
        // let mut packed_inp = Pointer::new(packed_inp_raw, packed_a_size);
        // #[cfg(not(feature = "bound_check"))]
        // let mut packed_inp = Pointer::new(packed_inp_raw);

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
                            // for _ in 0..kh {
                            //     for _ in 0..kw {
                            //         for _ in 0..icb {
                            //             let kernel_idx_2 = kernel_idx;
                            //             for mr in 0..owr {
                            //                 kernel_idx = kernel_idx_2;
                            //                 let a_val = packed_inp[idx as usize];
                            //                 idx += 1;
                            //                 // ocr is multiple of T::Vec::SIZE
                            //                 for nr in 0..ocr as i64 {
                            //                     let b_val = kernel[kernel_idx];
                            //                     out[(mr + kk + k) * osw + jj + j + nr] = a_val
                            //                         ._mul_add(
                            //                             b_val,
                            //                             out[(mr + kk + k) * osw + jj + j + nr],
                            //                         );
                            //                     kernel_idx += 1;
                            //                 }
                            //             }
                            //         }
                            //     }
                            // }
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

                            // for n in 0..kh {
                            //     let in_y = ll * step_height + n - ph_start;
                            //     if in_y < 0 || in_y >= img_height {
                            //         continue;
                            //     }
                            //     for m in 0..kw {
                            //         for ii in 0..icb {
                            //             let kernel_idx_2 = kernel_idx;
                            //             for mr in 0..owr {
                            //                 let in_x = (kk + k + mr) * step_width + m - pw_start;
                            //                 if in_x < 0 || in_x >= img_width {
                            //                     continue;
                            //                 }
                            //                 kernel_idx = kernel_idx_2;
                            //                 let a_val =
                            //                     T::Vec::splat(inp[n * ish + in_x * isw + i + ii]);
                            //                 // ocr is multiple of T::Vec::SIZE
                            //                 for nr in 0..NR as i64 {
                            //                     let b_val = unsafe {
                            //                         (kernel.ptr.add(
                            //                             kernel_idx as usize
                            //                                 + nr as usize * T::Vec::SIZE,
                            //                         )
                            //                             as *const <T as TypeCommon>::Vec)
                            //                             .read_unaligned()
                            //                     };
                            //                     out_regs[mr as usize][nr as usize] = a_val
                            //                         ._mul_add(
                            //                             b_val,
                            //                             out_regs[mr as usize][nr as usize],
                            //                         );
                            //                 }
                            //             }
                            //             kernel_idx += NR * T::Vec::SIZE as i64;
                            //         }
                            //     }
                            // }
                            // if i == 0 {
                            //     for mr in 0..owr {
                            //         let reg = out_regs[mr as usize].as_ptr() as *const T;
                            //         for nr in 0..ocr as i64 {
                            //             out[(mr + kk + k) * osw + jj + j + nr] =
                            //                 unsafe { reg.add(nr as usize).read() };
                            //         }
                            //     }
                            // } else {
                            //     for mr in 0..owr {
                            //         let reg = out_regs[mr as usize].as_ptr() as *const T;
                            //         for nr in 0..ocr as i64 {
                            //             let val = out[(mr + kk + k) * osw + jj + j + nr];
                            //             out[(mr + kk + k) * osw + jj + j + nr] =
                            //                 val._add(unsafe { reg.add(nr as usize).read() });
                            //         }
                            //     }
                            // }
                        }
                    }
                }
            }
        }

        // unsafe {
        //     std::alloc::dealloc(
        //         packed_inp_raw as *mut _,
        //         std::alloc::Layout::from_size_align(packed_a_size as usize, ALIGN).unwrap(),
        //     );
        // }
    });

    unsafe {
        std::alloc::dealloc(
            packed_kernel_raw as *mut _,
            std::alloc::Layout::from_size_align(packed_kernel_size as usize, ALIGN).unwrap(),
        );
    }
    Ok(output)
}

#[inline(always)]
fn mma<T: CommonBounds, const NR: usize, const MR: usize>(
    mut a: Pointer<T>,
    mut b: Pointer<T>,
    lda: i64,
    kc: usize,
    ks: i64,
    kh: usize,
    kw: usize,
) -> [[<T as TypeCommon>::Vec; NR]; MR] {
    let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); NR]; MR];
    for n in 0..kh {
        for m in 0..kw {
            for _ in 0..kc {
                for mr in 0..MR {
                    let a_vec = <T as TypeCommon>::Vec::splat(a[(mr as i64) * lda]);
                    for nr in 0..NR {
                        let b_vec = unsafe {
                            *(b.ptr.add(nr * <T as TypeCommon>::Vec::SIZE)
                                as *const <T as TypeCommon>::Vec)
                        };
                        c_local[mr][nr] = a_vec._mul_add(b_vec, c_local[mr][nr]);
                    }
                }
                b += NR * <T as TypeCommon>::Vec::SIZE;
                a += ks;
            }
        }
    }
    c_local
}

fn reorder_kernel<T: CommonBounds>(
    kernel: &Pointer<T>,
    reordered: Pointer<T>,
    jb: usize,
    [in_channel, ic_nvec]: [usize; 2],
    [out_channel, oc_nvec]: [usize; 2],
    [ks0, ks1, ks2]: [usize; 3],
    [kh, kw]: [usize; 2],
) {
    (0..in_channel)
        .into_par_iter()
        .step_by(T::Vec::SIZE * ic_nvec)
        .for_each(|ii| {
            let i_end = (ii + T::Vec::SIZE * ic_nvec).min(in_channel);
            let mut reordered = reordered.clone() + ii * out_channel * kh * kw;
            for jj in (0..out_channel).step_by(T::Vec::SIZE * oc_nvec * jb) {
                let jj_start = jj;
                let jj_end = (jj + T::Vec::SIZE * oc_nvec * jb).min(out_channel);
                let remain = (jj_end - jj_start) % (T::Vec::SIZE * oc_nvec);
                let oc_remain = remain % T::Vec::SIZE;
                for j in (jj_start..jj_end - remain).step_by(T::Vec::SIZE * oc_nvec) {
                    for n in 0..kh {
                        for m in 0..kw {
                            for i in ii..i_end {
                                for v in 0..oc_nvec {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(T::Vec::from_ptr(
                                            &kernel[i * ks2
                                                + n * ks0
                                                + m * ks1
                                                + j
                                                + v * T::Vec::SIZE],
                                        )); // prettier-ignore
                                    }
                                    reordered += T::Vec::SIZE;
                                }
                            }
                        }
                    }
                }
                if remain > 0 {
                    for j in (out_channel - remain..out_channel - oc_remain).step_by(T::Vec::SIZE) {
                        for n in 0..kh {
                            for m in 0..kw {
                                for i in ii..i_end {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(T::Vec::from_ptr(
                                            &kernel[n * ks0 + m * ks1 + i * ks2 + j],
                                        ));
                                    }
                                    reordered += T::Vec::SIZE;
                                }
                            }
                        }
                    }
                    for j in (out_channel - oc_remain..out_channel).step_by(T::Vec::SIZE) {
                        for n in 0..kh {
                            for m in 0..kw {
                                for i in ii..i_end {
                                    let ptr: *mut T = reordered.ptr;
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            &kernel[n * ks0 + m * ks1 + i * ks2 + j] as *const T,
                                            ptr,
                                            oc_remain,
                                        );
                                    }
                                    reordered += oc_remain;
                                }
                            }
                        }
                    }
                }
            }
        });
}

fn predict_ow_block(oc_block: usize) -> usize {
    REGNUM / (oc_block + 1)
}

/// calculate sub-optimal in channel block size and out channel block size,
/// to maximize the cache utilization and balance the memory access
fn kernel_params<T: CommonBounds>(
    out_channels: usize,
    in_channels: usize,
    ow_block: usize,
    oc_nvec: usize,
    oh_block: usize,
    [kh, kw]: [usize; 2],
    cache: Cache<T>,
) -> (usize, usize) {
    let l1 = cache.l1;
    let l2 = cache.l2;

    let ic_range = 1..(in_channels as usize) / T::Vec::SIZE;
    let jb_range = 1..(out_channels as usize) / (T::Vec::SIZE * oc_nvec);

    let best_params = ic_range
        .into_par_iter()
        .flat_map(|ic| jb_range.clone().into_par_iter().map(move |jb| (ic, jb)))
        .filter_map(|(ic, jb)| {
            let gemm_kernel_used = oc_nvec * T::Vec::SIZE * ic * T::Vec::SIZE;
            let gemm_inp_used = ow_block * ic * T::Vec::SIZE;
            let gemm_out_used = ow_block * oc_nvec * T::Vec::SIZE;
            let gemm_used = gemm_kernel_used + gemm_inp_used + gemm_out_used;
            let k_kernel_used = gemm_kernel_used * kh * kw;
            let k_inp_used = kh * kw * gemm_inp_used;
            let jb_k_kernel_used = jb * k_kernel_used;
            let jb_out_used = jb * gemm_out_used;
            let jb_inp_used = oh_block * k_inp_used;
            let total_used = jb_k_kernel_used + jb_out_used + jb_inp_used;

            if gemm_used <= l1 && total_used <= l2 {
                let balance = ((ic as f64) / (jb as f64)).max((jb as f64) / (ic as f64));
                let cache_utilization = (total_used as f64) / (l2 as f64);
                Some((ic, jb, balance, cache_utilization))
            } else {
                None
            }
        })
        .reduce(
            || (1, 1, f64::MAX, 0.0),
            |(best_ic, best_jb, best_balance, best_util), (ic, jb, balance, util)| {
                const BALANCE_WEIGHT: f64 = 0.6;
                const UTIL_WEIGHT: f64 = 0.4;

                let current_score = BALANCE_WEIGHT * (1.0 / balance) + UTIL_WEIGHT * util;
                let best_score = BALANCE_WEIGHT * (1.0 / best_balance) + UTIL_WEIGHT * best_util;

                if current_score > best_score {
                    (ic, jb, balance, util)
                } else {
                    (best_ic, best_jb, best_balance, best_util)
                }
            },
        );

    (best_params.0, best_params.1)
}

fn handle_remain<T: CommonBounds, F, F2, F3>(
    [jj_start, jj_end]: [i64; 2],
    [out_channels, oc_block_size]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    full_oc: F,
    one_oc: F2,
    partial_oc: F3,
) where
    F: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
    F2: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
    F3: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
{
    for j in (jj_start..jj_end - remain).step_by(oc_block_size as usize) {
        let original = kernel.clone();
        for l in ll..l_end {
            full_oc(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (i_end - ii) * (oc_block_size as i64);
    }
    let oc_remain = remain % (T::Vec::SIZE as i64);
    // loop over the remain part that are multiple of T::Vec::SIZE
    for j in (out_channels - remain..out_channels - oc_remain).step_by(T::Vec::SIZE) {
        let original = kernel.clone();
        for l in ll..l_end {
            one_oc(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (T::Vec::SIZE as i64) * (i_end - ii);
    }
    // loop over the remain part that are less than T::Vec::SIZE
    for j in (out_channels - oc_remain..out_channels).step_by(T::Vec::SIZE) {
        let original = kernel.clone();
        for l in ll..l_end {
            partial_oc(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * oc_remain * (i_end - ii);
    }
}

fn _handle_normal<T: CommonBounds, F>(
    [jj_start, jj_end]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    oc_block_size: usize,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    full_oc: F,
) where
    F: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
{
    for j in (jj_start..jj_end).step_by(oc_block_size) {
        let original = kernel.clone();
        for l in ll..l_end {
            full_oc(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (i_end - ii) * (oc_block_size as i64);
    }
}

fn handle_normal<T: CommonBounds, F>(
    [jj_start, jj_end]: [i64; 2],
    [out_width, ow_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    batch: i64,
    oc_block_size: usize,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    full_oc_kernel_fn: F,
    full_oc_kernel_ow_remain: F,
) where
    F: Fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
{
    let params = Params {
        arg1: [ii, i_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    };
    let out_width_full_end = out_width - (out_width % (ow_block as i64));
    let kernel_k = kernel.clone();
    // handle the out width full part
    for k in (0..out_width_full_end).step_by(ow_block as usize) {
        _handle_normal(
            [jj_start, jj_end],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            oc_block_size,
            out,
            kernel,
            |j, l, out, kernel| {
                full_oc_kernel_fn(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation,
                )
            },
        );
        *kernel = kernel_k.clone();
    }
    // handle the out width remain part
    for k in (out_width_full_end..out_width).step_by(ow_block as usize) {
        _handle_normal(
            [jj_start, jj_end],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            oc_block_size,
            out,
            kernel,
            |j, l, out, kernel| {
                full_oc_kernel_ow_remain(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation,
                )
            },
        );
        *kernel = kernel_k.clone();
    }
    *kernel += kernel_height * kernel_width * (jj_end - jj_start) * (i_end - ii);
}

fn with_bias_remain<T: CommonBounds>(
    [jj_start, jj_end]: [i64; 2],
    [out_channels, oc_block_size]: [i64; 2],
    [start, end, ow_block]: [i64; 3],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    bias_full_oc_kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    one_oc: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    partial_oc: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
) {
    let params = Params {
        arg1: [ii, i_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    };
    let partial_params = PartialParams {
        arg1: [ii, i_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
        oc_remain: remain % (T::Vec::SIZE as i64),
    };
    let kernel_k = kernel.clone();
    for k in (start..end).step_by(ow_block as usize) {
        handle_remain(
            [jj_start, jj_end],
            [out_channels, oc_block_size as i64],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            remain,
            out,
            kernel,
            |j, l, out, kernel| {
                bias_full_oc_kernel(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation,
                )
            },
            |j, l, out, kernel| {
                one_oc(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation,
                )
            },
            |j, l, out, kernel| {
                partial_oc(
                    PartialParams {
                        arg3: [batch, l, k, j],
                        ..partial_params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation,
                );
            },
        );
        *kernel = kernel_k.clone();
    }
}

fn with_bias_normal<T: CommonBounds>(
    [jj_start, jj_end]: [i64; 2],
    [start, end, ow_block]: [i64; 3],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    oc_block_size: i64,
    batch: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    bias_full_oc_kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
) {
    let params = Params {
        arg1: [ii, i_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    };
    let kernel_k = kernel.clone();
    for k in (start..end).step_by(ow_block as usize) {
        _handle_normal(
            [jj_start, jj_end],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            oc_block_size as usize,
            out,
            kernel,
            |j, l, out, kernel| {
                bias_full_oc_kernel(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation,
                )
            },
        );
        *kernel = kernel_k.clone();
    }
}

fn handle_bias_remain<T: CommonBounds>(
    [jj_start, jj_end]: [i64; 2],
    [out_channels, oc_block_size]: [i64; 2],
    [out_width, ow_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    bias_full_oc: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    partial_oc: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    bias_one_oc: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    bias_full_oc_ow_remain: Option<
        fn(
            Params,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec,
        ),
    >,
    bias_partial_oc_ow_remain: Option<
        fn(
            PartialParams,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec,
        ),
    >,
    bias_one_oc_ow_remain: Option<
        fn(
            Params,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec,
        ),
    >,
) {
    let out_width_full_end = out_width - (out_width % (ow_block as i64));
    with_bias_remain(
        [jj_start, jj_end],
        [out_channels, oc_block_size as i64],
        [0, out_width_full_end, ow_block as i64],
        [ll, l_end],
        [ii, i_end],
        [kernel_height, kernel_width],
        [osb, osh, osw],
        [isb, ish, isw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [img_height, img_width],
        batch,
        remain,
        out,
        kernel,
        &inp,
        &bias,
        activation,
        bias_full_oc,
        bias_one_oc,
        partial_oc,
    );
    // handle the out width remain part
    if let Some(full_oc_kernel_ow_remain) = &bias_full_oc_ow_remain {
        let one_oc_ow_remain = bias_one_oc_ow_remain.expect(&format!(
            "unable to find iconv2d_microkernel_{}x{}",
            ow_block, 1
        ));
        let partial_oc_ow_remain = bias_partial_oc_ow_remain
            .expect(&format!("unable to find oconv2d_microkernel_{}", ow_block));
        with_bias_remain(
            [jj_start, jj_end],
            [out_channels, oc_block_size as i64],
            [out_width_full_end, out_width, ow_block as i64],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            [osb, osh, osw],
            [isb, ish, isw],
            [step_height, step_width],
            [ph_start, pw_start],
            [dh, dw],
            [img_height, img_width],
            batch,
            remain,
            out,
            kernel,
            &inp,
            &bias,
            activation,
            *full_oc_kernel_ow_remain,
            one_oc_ow_remain,
            partial_oc_ow_remain,
        );
    }

    *kernel += kernel_height * kernel_width * (jj_end - jj_start) * (i_end - ii);
}

fn handle_bias_normal<T: CommonBounds>(
    [jj_start, jj_end]: [i64; 2],
    [out_width, ow_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    oc_block_size: i64,
    batch: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    bias_full_oc: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    bias_full_oc_ow_remain: Option<
        fn(
            Params,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec,
        ),
    >,
) {
    let out_width_full_end = out_width - (out_width % (ow_block as i64));
    with_bias_normal(
        [jj_start, jj_end],
        [0, out_width_full_end, ow_block as i64],
        [ll, l_end],
        [ii, i_end],
        [kernel_height, kernel_width],
        [osb, osh, osw],
        [isb, ish, isw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [img_height, img_width],
        oc_block_size,
        batch,
        out,
        kernel,
        &inp,
        &bias,
        activation,
        bias_full_oc,
    );
    // handle the out width remain part
    if let Some(full_oc_kernel_ow_remain) = &bias_full_oc_ow_remain {
        with_bias_normal(
            [jj_start, jj_end],
            [out_width_full_end, out_width, ow_block as i64],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            [osb, osh, osw],
            [isb, ish, isw],
            [step_height, step_width],
            [ph_start, pw_start],
            [dh, dw],
            [img_height, img_width],
            oc_block_size,
            batch,
            out,
            kernel,
            &inp,
            &bias,
            activation,
            *full_oc_kernel_ow_remain,
        );
    }

    *kernel += kernel_height * kernel_width * (jj_end - jj_start) * (i_end - ii);
}

fn with_normal_remain<T: CommonBounds>(
    [jj_start, jj_end]: [i64; 2],
    [out_channels, oc_block_size]: [i64; 2],
    [start, end, ow_block]: [i64; 3],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    bias_full_oc_kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    one_oc: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
    partial_oc: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
) {
    let params = Params {
        arg1: [ii, i_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    };
    let partial_params = PartialParams {
        arg1: [ii, i_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
        oc_remain: remain % (T::Vec::SIZE as i64),
    };
    let kernel_k = kernel.clone();
    for k in (start..end).step_by(ow_block as usize) {
        handle_remain(
            [jj_start, jj_end],
            [out_channels, oc_block_size as i64],
            [ll, l_end],
            [ii, i_end],
            [kernel_height, kernel_width],
            remain,
            out,
            kernel,
            |j, l, out, kernel| {
                bias_full_oc_kernel(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation,
                )
            },
            |j, l, out, kernel| {
                one_oc(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation,
                )
            },
            |j, l, out, kernel| {
                partial_oc(
                    PartialParams {
                        arg3: [batch, l, k, j],
                        ..partial_params
                    },
                    out,
                    kernel,
                    &inp,
                    activation,
                );
            },
        );
        *kernel = kernel_k.clone();
    }
}

fn handle_normal_remain<T: CommonBounds>(
    [jj_start, jj_end]: [i64; 2],
    [out_channels, oc_block_size]: [i64; 2],
    [out_width, ow_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [img_height, img_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
    full_oc: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
    partial_oc: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    one_oc: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
    full_oc_ow_remain: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    partial_oc_ow_remain: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
    one_oc_ow_remain: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
) {
    let out_width_full_end = out_width - (out_width % (ow_block as i64));
    with_normal_remain(
        [jj_start, jj_end],
        [out_channels, oc_block_size as i64],
        [0, out_width_full_end, ow_block as i64],
        [ll, l_end],
        [ii, i_end],
        [kernel_height, kernel_width],
        [osb, osh, osw],
        [isb, ish, isw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [img_height, img_width],
        batch,
        remain,
        out,
        kernel,
        &inp,
        activation,
        full_oc,
        one_oc,
        partial_oc,
    );
    // handle the out width remain part
    with_normal_remain(
        [jj_start, jj_end],
        [out_channels, oc_block_size as i64],
        [out_width_full_end, out_width, ow_block as i64],
        [ll, l_end],
        [ii, i_end],
        [kernel_height, kernel_width],
        [osb, osh, osw],
        [isb, ish, isw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [img_height, img_width],
        batch,
        remain,
        out,
        kernel,
        &inp,
        activation,
        full_oc_ow_remain,
        one_oc_ow_remain,
        partial_oc_ow_remain,
    );
    *kernel += kernel_height * kernel_width * (jj_end - jj_start) * (i_end - ii);
}

// #[track_caller]
// pub(crate) fn diff_conv2d<T: CommonBounds, const DEVICE: usize>(
//     input: &DiffTensor<T, Cpu, DEVICE>,
//     kernels: &DiffTensor<T, Cpu, DEVICE>,
//     bias: Option<&DiffTensor<T, Cpu, DEVICE>>,
//     steps: [i64; 2],
//     padding: [(i64, i64); 2],
//     dilation: [i64; 2],
// ) -> Result<DiffTensor<T, Cpu, DEVICE>, TensorError> {
//     let res = input.inner.conv2d(
//         &kernels.inner,
//         bias.map(|b| &b.inner),
//         steps,
//         padding,
//         dilation,
//         None,
//     )?;
//     let mut kernel = kernels.clone();
//     let mut bias = bias.map(|b| b.clone());
//     let mut operand = input.clone();
//     Ok(DiffTensor {
//         inner: res,
//         grad: Rc::new(RefCell::new(None)),
//         out_degree: Rc::new(RefCell::new(0)),
//         backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
//             let input_grad = grad.conv2d_transpose(
//                 &kernel.inner,
//                 None,
//                 steps,
//                 padding,
//                 [0, 0], // output_padding = 0
//                 dilation,
//             )?;
//             let kernel_grad = conv2d_backward_kernel(
//                 &operand.inner.inner,
//                 &grad.inner,
//                 &kernel.inner.shape()[2..],
//                 steps,
//                 padding,
//             )?;
//             if let Some(bias) = &mut bias {
//                 let bias_grad = conv2d_backward_bias(&grad.inner)?;
//                 handle_grad(bias, bias_grad.into(), &[])?;
//             }
//             handle_grad(&mut operand, kernel_grad.into(), &[])?;
//             handle_grad(&mut kernel, input_grad.into(), &[])?;
//             Ok(false)
//         })),
//     })
// }
