use gemm_common::cache::{DivCeil, KernelParams, CACHE_INFO};
use hpt_common::Pointer;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;
use hpt_types::{into_scalar::Cast, type_promote::NormalOutPromote};
use num::integer::gcd;

use crate::ALIGN;

pub(crate) fn img2col_nhwc<T: CommonBounds>(
    mut buffer: Pointer<T>,
    in_ptr: Pointer<T>,
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
    for p in 0..out_height {
        for q in 0..out_width {
            for m in 0..kh {
                let in_y = p * stride_h + m * dilation_h - pad_h;
                for n in 0..kw {
                    let in_x = q * stride_w + n * dilation_w - pad_w;

                    let offset = in_y * h_stride + in_x * w_stride;
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

pub(crate) fn pack_kernel<T: CommonBounds>(
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

pub(crate) fn pack_kernel_mp<T: CommonBounds>(
    mut packed_kernel: Pointer<<T as NormalOutPromote>::Intermediate>,
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
                                    kernel[n * ks0 + m * ks1 + (i + ii) * ks2 + jj + j + nr].cast();
                                idx += 1;
                            }
                            for _ in ocr..or {
                                packed_kernel[idx] = <T as NormalOutPromote>::Intermediate::ZERO;
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn calculate_kernel_params<T: CommonBounds>(
    in_channels: i64,
    out_channels: i64,
    out_width: i64,
    mr: usize,
    nr: usize,
) -> KernelParams {
    let mut param = if in_channels <= 64 && out_channels <= 64 {
        // skip expensive kernel_params call for small sizes
        let kc = in_channels.min(512);
        let alloc = CACHE_INFO[1].cache_bytes / core::mem::size_of::<T>();
        let nc = (alloc / (kc as usize) / (nr as usize)) * (nr as usize);
        KernelParams {
            kc: kc as usize,
            mc: (out_width as usize).msrv_next_multiple_of(mr),
            nc,
        }
    } else {
        gemm_common::cache::kernel_params(
            out_channels as usize,
            out_width as usize,
            in_channels as usize,
            nr,
            mr,
            std::mem::size_of::<T>(),
        )
    };
    if param.nc == 0 {
        param.nc = (out_channels as usize).msrv_next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = (out_width as usize).msrv_next_multiple_of(mr);
    }

    param
}

pub(crate) fn create_packed_kernel<T: CommonBounds>(
    kh: i64,
    kw: i64,
    in_channels: i64,
    out_channels: i64,
    oc: i64,
    nr: i64,
) -> (Pointer<T>, std::alloc::Layout) {
    let packed_kernel_size = kh
        * kw
        * in_channels
        * ((out_channels as usize).div_ceil(oc as usize) as i64)
        * ((oc as usize).div_ceil(nr as usize) as i64)
        * (nr as i64)
        * (std::mem::size_of::<T>() as i64);

    let layout = std::alloc::Layout::from_size_align(packed_kernel_size as usize, ALIGN).unwrap();
    let packed_kernel_raw = unsafe { std::alloc::alloc(layout) };
    #[cfg(feature = "bound_check")]
    let packed_kernel = Pointer::new(
        packed_kernel_raw as *mut T,
        packed_kernel_size / (std::mem::size_of::<T>() as i64),
    );
    #[cfg(not(feature = "bound_check"))]
    let packed_kernel = Pointer::new(packed_kernel_raw as *mut T);

    (packed_kernel, layout)
}

pub(crate) fn create_packed_input_img2col<T: CommonBounds>(
    kh: i64,
    kw: i64,
    in_channels: i64,
    out_height: i64,
    out_width: i64,
) -> (Pointer<T>, std::alloc::Layout) {
    let packed_size =
        kh * kw * in_channels * out_height * out_width * std::mem::size_of::<T>() as i64;

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
