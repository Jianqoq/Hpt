use std::cell::OnceCell;

use gemm_common::cache::{DivCeil, KernelParams};
use hpt_common::Pointer;
use hpt_common::error::base::TensorError;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::{DType, ToDType, TypeCommon};
use hpt_types::type_promote::NormalOut;
use hpt_types::{into_scalar::Cast, type_promote::NormalOutPromote};
use num::integer::gcd;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::ops::tensor::binary::binary_fn_with_out;
use crate::ops::tensor::unary::unary_fn_with_out;
use crate::{ALIGN, Device, Tensor};

pub(crate) fn img2col_nhwc<T: CommonBounds>(
    mut buffer: Pointer<T>,
    in_ptr: Pointer<T>,
    batch: i64,
    in_strides: &[i64],
    in_height: i64,
    in_width: i64,
    in_channels: i64,
    out_height: i64,
    out_width: i64,
    [kh, kw]: [i64; 2],
    [stride_h, stride_w]: [i64; 2],
    [pad_h, pad_w]: [i64; 2],
    [dilation_h, dilation_w]: [i64; 2],
) {
    let mut buffer_idx: i64 = 0;
    let h_stride = in_strides[1];
    let w_stride = in_strides[2];
    let batch_stride = in_strides[0];
    for b in 0..batch {
        for p in 0..out_height {
            for q in 0..out_width {
                for m in 0..kh {
                    let in_y = p * stride_h + m * dilation_h - pad_h;
                    for n in 0..kw {
                        let in_x = q * stride_w + n * dilation_w - pad_w;

                        let offset = in_y * h_stride + in_x * w_stride + batch_stride * b;
                        if in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width {
                            for i in 0..in_channels {
                                buffer[buffer_idx] = in_ptr[offset + i];
                                buffer_idx += 1;
                            }
                        } else {
                            for _ in 0..in_channels {
                                buffer[buffer_idx] = T::ZERO;
                                buffer_idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn pack_kernel<T: CommonBounds>(
    packed_kernel: Pointer<T>,
    kernel: Pointer<T>,
    in_channels: i64,
    out_channels: i64,
    ic: i64,
    oc: i64,
    or: i64,
    [kh, kw]: [i64; 2],
    [ks0, ks1, ks2, ks3]: [i64; 4],
) {
    use hpt_types::traits::VecTrait;
    let num_vec = or / (T::Vec::SIZE as i64);
    fn calculate_block_size(icb: i64, ocb: i64, or: i64, kh: i64, kw: i64) -> i64 {
        let mut size = 0;
        for jj in (0..ocb).step_by(or as usize) {
            let ocr = or.min(ocb - jj);
            size += kh * kw * icb * (ocr + (or - ocr));
        }
        size
    }

    let mut work_items = Vec::new();
    let mut total_offset = 0;

    for i in (0..in_channels).step_by(ic as usize) {
        let icb = ic.min(in_channels - i);
        for j in (0..out_channels).step_by(oc as usize) {
            let ocb = oc.min(out_channels - j);
            let block_size = calculate_block_size(icb, ocb, or, kh, kw);

            work_items.push((i, j, icb, ocb, total_offset));
            total_offset += block_size;
        }
    }

    work_items.par_iter().for_each(|&(i, j, icb, ocb, offset)| {
        let mut local_idx = offset;
        let mut packed_kernel = packed_kernel;
        for jj in (0..ocb).step_by(or as usize) {
            let ocr = or.min(ocb - jj);
            if ocr == or && ks3 == 1 {
                for n in 0..kh {
                    for m in 0..kw {
                        for ii in 0..icb {
                            unsafe {
                                let ptr = kernel
                                    .ptr
                                    .offset((n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j) as isize)
                                    as *const T::Vec;
                                let packed_ptr =
                                    packed_kernel.ptr.offset(local_idx as isize) as *mut T::Vec;
                                for nr in 0..num_vec {
                                    packed_ptr
                                        .offset(nr as isize)
                                        .write_unaligned(ptr.offset(nr as isize).read_unaligned());
                                }
                                local_idx += num_vec * (T::Vec::SIZE as i64);
                            }
                        }
                    }
                }
            } else {
                for n in 0..kh {
                    for m in 0..kw {
                        for ii in 0..icb {
                            for nr in 0..ocr {
                                packed_kernel[local_idx] =
                                    kernel[n * ks0 + m * ks1 + (i + ii) * ks2 + (jj + j + nr) * ks3];
                                local_idx += 1;
                            }
                            for _ in ocr..or {
                                packed_kernel[local_idx] = T::ZERO;
                                local_idx += 1;
                            }
                        }
                    }
                }
            }
        }
    });
}

pub(crate) fn pack_kernel_mp<T: CommonBounds>(
    packed_kernel: Pointer<<T as NormalOutPromote>::Intermediate>,
    kernel: Pointer<T>,
    in_channels: i64,
    out_channels: i64,
    ic: i64,
    oc: i64,
    or: i64,
    [kh, kw]: [i64; 2],
    [ks0, ks1, ks2, ks3]: [i64; 4],
) where
    T: Cast<<T as NormalOutPromote>::Intermediate>,
    <T as NormalOutPromote>::Intermediate: CommonBounds,
{
    fn calculate_block_size(icb: i64, ocb: i64, or: i64, kh: i64, kw: i64) -> i64 {
        let mut size = 0;
        for jj in (0..ocb).step_by(or as usize) {
            let ocr = or.min(ocb - jj);
            size += kh * kw * icb * (ocr + (or - ocr));
        }
        size
    }

    let mut work_items = Vec::new();
    let mut total_offset = 0;

    for i in (0..in_channels).step_by(ic as usize) {
        let icb = ic.min(in_channels - i);
        for j in (0..out_channels).step_by(oc as usize) {
            let ocb = oc.min(out_channels - j);
            let block_size = calculate_block_size(icb, ocb, or, kh, kw);

            work_items.push((i, j, icb, ocb, total_offset));
            total_offset += block_size;
        }
    }

    work_items.par_iter().for_each(|&(i, j, icb, ocb, offset)| {
        let mut local_idx = offset;
        let mut packed_kernel = packed_kernel;

        for jj in (0..ocb).step_by(or as usize) {
            let ocr = or.min(ocb - jj);
            for n in 0..kh {
                for m in 0..kw {
                    for ii in 0..icb {
                        for nr in 0..ocr {
                            packed_kernel[local_idx] = kernel
                                [n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr * ks3]
                                .cast();
                            local_idx += 1;
                        }
                        for _ in ocr..or {
                            packed_kernel[local_idx] = <T as NormalOutPromote>::Intermediate::ZERO;
                            local_idx += 1;
                        }
                    }
                }
            }
        }
    });
}

pub(crate) fn calculate_kernel_params<T: CommonBounds>(
    in_channels: i64,
    out_channels: i64,
    out_width: i64,
    mr: usize,
    nr: usize,
    [kh, kw]: [usize; 2],
) -> KernelParams {
    let mut param = kernel_params(
        out_channels as usize,
        out_width as usize,
        in_channels as usize,
        nr,
        mr,
        std::mem::size_of::<T>(),
        [kh, kw],
    );
    if param.nc == 0 {
        param.nc = (out_channels as usize).msrv_next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = (out_width as usize).msrv_next_multiple_of(mr);
    }

    param
}

pub(crate) fn create_packed_kernel(
    dtype: DType,
    device: Device,
    kh: i64,
    kw: i64,
    in_channels: i64,
    out_channels: i64,
    oc: i64,
    nr: i64,
) -> Result<Tensor, TensorError> {
    let packed_kernel_size = kh
        * kw
        * in_channels
        * ((out_channels as usize).div_ceil(oc as usize) as i64)
        * ((oc as usize).div_ceil(nr as usize) as i64)
        * (nr as i64);

    let buffer = Tensor::empty(&[packed_kernel_size], dtype, device)?;

    Ok(buffer)
}

pub(crate) fn create_packed_input_img2col<T: CommonBounds>(
    batch: i64,
    kh: i64,
    kw: i64,
    in_channels: i64,
    out_height: i64,
    out_width: i64,
) -> (Pointer<T>, std::alloc::Layout) {
    let packed_size =
        batch * kh * kw * in_channels * out_height * out_width * (std::mem::size_of::<T>() as i64);

    let layout = std::alloc::Layout::from_size_align(packed_size as usize, ALIGN).unwrap();
    let buffer = unsafe { std::alloc::alloc(layout) };
    let buffer_ptr = Pointer::new(
        buffer as *mut T,
        packed_size / (std::mem::size_of::<T>() as i64),
    );

    (buffer_ptr, layout)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CacheInfo {
    pub(crate) bytes: usize,
    pub(crate) associativity: usize,
    pub(crate) cache_line_bytes: usize,
}

const CACHE_INFO: OnceCell<[CacheInfo; 3]> = OnceCell::new();

pub(crate) fn get_cache_info() -> [CacheInfo; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::{ CacheType, CpuId };
        *CACHE_INFO.get_or_init(|| {
            let cpuid = CpuId::new();
            let mut cache_info = [
                CacheInfo {
                    bytes: 0,
                    associativity: 0,
                    cache_line_bytes: 0,
                };
                3
            ];
            if let Some(cparams) = cpuid.get_cache_parameters() {
                for cache in cparams {
                    let size =
                        cache.associativity() *
                        cache.physical_line_partitions() *
                        cache.coherency_line_size() *
                        cache.sets();
                    let valid_cache =
                        (cache.cache_type() == CacheType::Data ||
                            cache.cache_type() == CacheType::Unified) && cache.level() <= 3;
                    if valid_cache {
                        let info = CacheInfo {
                            bytes: size,
                            associativity: cache.associativity(),
                            cache_line_bytes: cache.coherency_line_size(),
                        };
                        cache_info[(cache.level() as usize) - 1] = info;
                    }
                }
            } else {
                panic!("No cache parameter information available");
            }
            cache_info
        })
    }
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        *CACHE_INFO.get_or_init(|| {
            let mut cache_info = [
                CacheInfo {
                    bytes: 0,
                    associativity: 0,
                    cache_line_bytes: 0,
                };
                3
            ];
            for level in 1..=3 {
                let mut size: u64 = 0;
                let mut line_size: u64 = 0;

                let mut size_len = std::mem::size_of::<u64>();
                let mut line_size_len = std::mem::size_of::<u64>();

                let name = if level == 1 {
                    "hw.l1dcachesize"
                } else if level == 2 {
                    "hw.l2cachesize"
                } else {
                    "hw.l3cachesize"
                };
                unsafe {
                    libc::sysctlbyname(
                        CString::new(name).unwrap().as_ptr(),
                        &mut size as *mut _ as *mut libc::c_void,
                        &mut size_len,
                        std::ptr::null_mut(),
                        0
                    );
                }

                unsafe {
                    libc::sysctlbyname(
                        CString::new("hw.cachelinesize").unwrap().as_ptr(),
                        &mut line_size as *mut _ as *mut libc::c_void,
                        &mut line_size_len,
                        std::ptr::null_mut(),
                        0
                    );
                }
                cache_info[level - 1] = CacheInfo {
                    bytes: size as usize,
                    cache_line_bytes: line_size as usize,
                    associativity: 8,
                };
            }
            cache_info
        })
    }
}

// code is from gemm-common
pub fn _kernel_params(
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    sizeof: usize
) -> KernelParams {
    #[inline]
    fn round_down(a: usize, b: usize) -> usize {
        (a / b) * b
    }
    if m == 0 || n == 0 || k == 0 {
        return KernelParams {
            kc: k,
            mc: m,
            nc: n,
        };
    }

    let info = get_cache_info();

    let l1_cache_bytes = info[0].bytes.max(32 * 1024);
    let l2_cache_bytes = info[1].bytes;
    let l3_cache_bytes = info[2].bytes;

    let l1_line_bytes = info[0].cache_line_bytes.max(64);

    let l1_assoc = info[0].associativity.max(3);
    let l2_assoc = info[1].associativity.max(3);
    let l3_assoc = info[2].associativity.max(3);

    let l1_n_sets = l1_cache_bytes / (l1_line_bytes * l1_assoc);

    // requires
    // A micropanels must occupy different cache sets
    // so that loading a micropanel evicts the previous one
    // => byte stride must be multiple of n_sets×line_bytes
    //
    // => mr×kc×scalar_bytes == C_A × l1_line_bytes × l1_n_sets
    //
    // l1 must be able to hold A micropanel, B micropanel
    //
    // => C_A + C_B <= l1_assoc

    // a×n = b×m
    // find lcm of a, b
    // n = lcm / a = b/gcd(a,b)
    // m = lcm / b = a/gcd(a,b)

    let gcd = gcd(mr * sizeof, l1_line_bytes * l1_n_sets);
    let kc_0 = (l1_line_bytes * l1_n_sets) / gcd;
    let c_lhs = (mr * sizeof) / gcd;
    let c_rhs = (nr * kc_0 * sizeof) / (l1_line_bytes * l1_n_sets);
    let kc_multiplier = l1_assoc / (c_lhs + c_rhs);
    // let auto_kc = kc_0 * kc_multiplier;
    let auto_kc = (kc_0 * kc_multiplier.next_power_of_two()).max(512).min(k);
    let k_iter = k.div_ceil(auto_kc);
    let auto_kc = k.div_ceil(k_iter);

    // l2 cache must hold
    //  - B micropanel: nr×kc: assume 1 assoc degree
    //  - A macropanel: mc×kc
    // mc×kc×scalar_bytes
    let auto_mc = if l2_cache_bytes == 0 {
        panic!();
    } else {
        let rhs_micropanel_bytes = nr * auto_kc * sizeof;
        let rhs_l2_assoc = rhs_micropanel_bytes.div_ceil(l2_cache_bytes / l2_assoc);
        let lhs_l2_assoc = (l2_assoc - 1 - rhs_l2_assoc).max(1);

        let mc_from_lhs_l2_assoc = |lhs_l2_assoc: usize| -> usize {
            (lhs_l2_assoc * l2_cache_bytes) / (l2_assoc * sizeof * auto_kc)
        };

        let auto_mc = round_down(mc_from_lhs_l2_assoc(lhs_l2_assoc), mr);
        let m_iter = m.div_ceil(auto_mc);
        m.div_ceil(m_iter * mr) * mr
    };
    let auto_mc = Ord::min(auto_mc, 8 * mr);

    // l3 cache must hold
    //  - A macropanel: mc×kc: assume 1 assoc degree
    //  - B macropanel: nc×kc
    let auto_nc = if l3_cache_bytes == 0 {
        0
    } else {
        // let lhs_macropanel_bytes = auto_mc * auto_kc * sizeof;
        // let lhs_l3_assoc = msrv_div_ceil(lhs_macropanel_bytes, l3_cache_bytes / l3_assoc);
        let rhs_l3_assoc = l3_assoc - 1;
        let rhs_macropanel_max_bytes = (rhs_l3_assoc * l3_cache_bytes) / l3_assoc;

        let auto_nc = round_down(rhs_macropanel_max_bytes / (sizeof * auto_kc), nr);
        let n_iter = n.div_ceil(auto_nc);
        n.div_ceil(n_iter * nr) * nr
    };

    KernelParams {
        kc: auto_kc,
        mc: auto_mc,
        nc: auto_nc,
    }
}

/// cache block calculation based on [gemm](https://github.com/sarah-quinones/gemm)
pub(crate) fn kernel_params(
    n: usize,
    m: usize,
    k: usize,
    nr: usize,
    mr: usize,
    sizeof: usize,
    _: [usize; 2],
) -> KernelParams {
    _kernel_params(n, m, k, nr, mr, sizeof)
}

#[inline]
pub(crate) fn handle_post<T: CommonBounds + ToDType>(
    output: &mut Tensor,
    bias: Option<&Tensor>,
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(T::Vec) -> T::Vec>,
) -> Result<(), TensorError> {
    match (bias, post_scalar, post_vec) {
        (None, None, None) => {}
        (None, Some(post_scalar), Some(post_vec)) => {
            unary_fn_with_out(&output, post_vec, post_scalar, Some(output.clone()))?;
        }
        (Some(bias), None, None) => {
            output.add_(bias, &mut output.clone())?;
        }
        (Some(bias), Some(post_scalar), Some(post_vec)) => {
            binary_fn_with_out(
                &output,
                &bias,
                |lhs: T, rhs: T| post_scalar(lhs._add(rhs)),
                |lhs: T::Vec, rhs: T::Vec| post_vec(lhs._add(rhs)),
                Some(output.clone()),
            )?;
        }
        _ => {
            unreachable!();
        }
    }
    Ok(())
}

pub(crate) fn cal_conv2d_output_shape(
    img_height: i64,
    img_width: i64,
    kh: i64,
    kw: i64,
    padding: &[(i64, i64); 2],
    stride: &[i64; 2],
    dilation: &[i64; 2],
) -> (i64, i64) {
    let out_height =
        (img_height + padding[0].0 + padding[0].1 - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
    let out_width =
        (img_width + padding[1].0 + padding[1].1 - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
    (out_height, out_width)
}
