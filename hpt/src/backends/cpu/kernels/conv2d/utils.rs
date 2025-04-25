use gemm_common::cache::{DivCeil, KernelParams, CACHE_INFO};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::Pointer;
use hpt_traits::ops::binary::NormalBinOps;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;
use hpt_types::type_promote::NormalOut;
use hpt_types::{into_scalar::Cast, type_promote::NormalOutPromote};
use num::integer::gcd;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::backends::cpu::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::backends::cpu::utils::unary::unary::unary_fn_with_out;
use crate::tensor_base::_Tensor;
use crate::ALIGN;

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
    [ks0, ks1, ks2]: [i64; 3],
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
            if ocr == or {
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
                                    kernel[n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr];
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
    [ks0, ks1, ks2]: [i64; 3],
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
                            packed_kernel[local_idx] =
                                kernel[n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr].cast();
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

pub(crate) fn create_packed_kernel<T: CommonBounds, const DEVICE: usize, A>(
    kh: i64,
    kw: i64,
    in_channels: i64,
    out_channels: i64,
    oc: i64,
    nr: i64,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let packed_kernel_size = kh
        * kw
        * in_channels
        * ((out_channels as usize).div_ceil(oc as usize) as i64)
        * ((oc as usize).div_ceil(nr as usize) as i64)
        * (nr as i64);

    let buffer = _Tensor::<T, Cpu, DEVICE, A>::empty(&[packed_kernel_size as usize])?;

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
    #[cfg(feature = "bound_check")]
    let buffer_ptr = Pointer::new(
        buffer as *mut T,
        packed_size / (std::mem::size_of::<T>() as i64),
    );
    #[cfg(not(feature = "bound_check"))]
    let buffer_ptr = Pointer::new(buffer as *mut T);

    (buffer_ptr, layout)
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
    fn round_down(a: usize, b: usize) -> usize {
        (a / b) * b
    }
    if n == 0 || m == 0 || k == 0 {
        return KernelParams {
            kc: k,
            mc: n,
            nc: m,
        };
    }

    let info = *CACHE_INFO;

    let l1_cache_bytes = info[0].cache_bytes.max(32 * 1024);
    let l2_cache_bytes = info[1].cache_bytes;
    let l3_cache_bytes = info[2].cache_bytes;

    let l1_line_bytes = info[0].cache_line_bytes.max(64);

    let l1_assoc = info[0].associativity.max(2);
    let l2_assoc = info[1].associativity.max(2);
    let l3_assoc = info[2].associativity.max(2);

    let l1_n_sets = l1_cache_bytes / (l1_line_bytes * l1_assoc);

    let gcd = gcd(nr * sizeof, l1_line_bytes * l1_n_sets);
    let kc_0 = (l1_line_bytes * l1_n_sets) / gcd; // maximum # of nr * sizeof access that has no conflicts
    let c_rhs = (nr * kc_0 * sizeof).next_multiple_of(l1_line_bytes) / (l1_line_bytes * l1_n_sets);
    let c_lhs =
        (mr * (kc_0 * sizeof).next_multiple_of(l1_line_bytes)) / (l1_line_bytes * l1_n_sets);
    let kc_multiplier = l1_assoc / (c_rhs + c_lhs);
    let auto_kc = (kc_0 * kc_multiplier.max(1))
        .next_power_of_two()
        .max(512)
        .min(k);
    let k_iter = k.div_ceil(auto_kc);
    let auto_kc = k.div_ceil(k_iter);

    let auto_nc = if l2_cache_bytes == 0 {
        panic!();
    } else {
        let lhs_micropanel_bytes = mr * (auto_kc * sizeof).next_multiple_of(l1_line_bytes);
        let lhs_l2_assoc = lhs_micropanel_bytes.div_ceil(l2_cache_bytes / l2_assoc);
        let rhs_l2_assoc = (l2_assoc - lhs_l2_assoc).max(1);

        let nc_from_rhs_l2_assoc = |rhs_l2_assoc: usize| -> usize {
            (rhs_l2_assoc * l2_cache_bytes) / (l2_assoc * sizeof * auto_kc)
        };

        let auto_nc = round_down(nc_from_rhs_l2_assoc(rhs_l2_assoc), nr);
        let n_iter = n.div_ceil(auto_nc);
        n.div_ceil(n_iter * nr) * nr
    };
    let auto_nc = Ord::min(auto_nc, 4 * nr);

    let auto_mc = if l3_cache_bytes == 0 {
        0
    } else {
        let rhs_l3_assoc = l3_assoc - 1;
        let rhs_macropanel_max_bytes = (rhs_l3_assoc * l3_cache_bytes) / l3_assoc;

        let auto_nc = round_down(rhs_macropanel_max_bytes / (sizeof * auto_kc), mr);
        let n_iter = m.div_ceil(auto_nc);
        m.div_ceil(n_iter * mr) * mr
    };

    KernelParams {
        kc: auto_kc,
        mc: auto_mc,
        nc: auto_nc,
    }
}

pub(crate) fn handle_post<T: CommonBounds, const DEVICE: usize, A>(
    output: &mut _Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(T::Vec) -> T::Vec>,
) -> Result<(), TensorError>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    match (bias, post_scalar, post_vec) {
        (None, None, None) => {}
        (None, Some(post_scalar), Some(post_vec)) => {
            unary_fn_with_out(&output, post_vec, post_scalar, Some(output.clone()))?;
        }
        (Some(bias), None, None) => {
            output.add_(bias, &mut output.clone())?;
        }
        (Some(bias), Some(post_scalar), Some(post_vec)) => {
            binary_fn_with_out_simd(
                &output,
                &bias,
                |lhs, rhs| post_scalar(lhs._add(rhs)),
                |lhs, rhs| post_vec(lhs._add(rhs)),
                Some(output.clone()),
            )?;
        }
        _ => {
            unreachable!();
        }
    }
    Ok(())
}
