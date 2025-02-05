use crate::CommonBounds;
use duplicate::duplicate_item;
use tensor_common::Pointer;
use tensor_macros::conv2d_microkernel_declare_const;
use tensor_types::into_scalar::Cast;
use tensor_types::traits::VecTrait;
use tensor_types::type_promote::NormalOut;

pub(crate) struct Params {
    pub(crate) arg1: [i64; 2],
    pub(crate) arg2: [i64; 2],
    pub(crate) arg3: [i64; 4],
    pub(crate) arg4: [i64; 3],
    pub(crate) arg5: [i64; 2],
    pub(crate) arg6: [i64; 3],
    pub(crate) pads: [i64; 2],
    pub(crate) arg8: [i64; 2],
    pub(crate) arg9: [i64; 2],
}

macro_rules! repeat_kernel {
    ($name:ident, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                unsafe {
                    T::Vec::from_ptr(&$name[$idx * T::Vec::SIZE])
                 },
            )*)
        }
    };
}

macro_rules! repeat_out_vecs {
    ($name:ident, $common_idx:expr, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                unsafe {
                    T::Vec::from_ptr(&$name[$common_idx + $idx * T::Vec::SIZE as i64])
                 },
            )*)
        }
    };
}

macro_rules! repeat_mul_add {
    ($inp:expr, $krs:expr, $out_vecs:expr, [$($idx:expr),*]) => {
        paste::paste! {
            $(
                $out_vecs.$idx = $inp._mul_add($krs.$idx, $out_vecs.$idx);
            )*
        }
    };
}

macro_rules! repeat_write {
    ($out:expr, $out_vecs:expr, $out_idx_common:expr, [$($idx:expr),*]) => {
        paste::paste! {
            $(
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        $out_vecs.$idx.as_ptr(),
                        &mut $out[$out_idx_common + ($idx * T::Vec::SIZE) as i64],
                        T::Vec::SIZE,
                    )
                };
            )*
        }
    };
}

#[duplicate_item(
    template_function       repeats;
    [micro_kernel_5x1]      [[0]];
    [micro_kernel_4x1]      [[0]];
    [micro_kernel_3x1]      [[0]];
    [micro_kernel_2x1]      [[0]];
    [micro_kernel_1x1]      [[0]];
    [micro_kernel_5x2]      [[0, 1]];
    [micro_kernel_4x2]      [[0, 1]];
    [micro_kernel_3x2]      [[0, 1]];
    [micro_kernel_2x2]      [[0, 1]];
    [micro_kernel_1x2]      [[0, 1]];
    [micro_kernel_5x4]      [[0, 1, 2, 3]];
    [micro_kernel_4x4]      [[0, 1, 2, 3]];
    [micro_kernel_3x4]      [[0, 1, 2, 3]];
    [micro_kernel_2x4]      [[0, 1, 2, 3]];
    [micro_kernel_1x4]      [[0, 1, 2, 3]];
    [micro_kernel_5x8]      [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_4x8]      [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_3x8]      [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_2x8]      [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_1x8]      [[0, 1, 2, 3, 4, 5, 6, 7]];
)]
pub(crate) fn template_function<T: CommonBounds>(
    params: Params,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: Cast<T>,
{
    let Params {
        arg1: [oo, o_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    const IC_NVEC: usize = OC_BLOCK;
    const IW_BLOCK: usize = OW_BLOCK;
    for n in 0..kh {
        let h_out = l * step_height + n * dh - ph_start;
        let h_in_range = h_out >= 0 && h_out < out_height;
        if h_in_range {
            for m in 0..kw {
                for o in oo..o_end {
                    let krs = repeat_kernel!(kernel, repeats);
                    for kk in 0..IW_BLOCK as i64 {
                        let w_out = (k + kk) * step_width + m * dw - pw_start;
                        if w_out >= 0 && w_out < out_width {
                            let inp_idx = b * osb + l * osh + (k + kk) * osw + o;
                            let inp_vec = T::Vec::splat(inp[inp_idx]);
                            let out_idx_common = b * isb + h_out * ish + w_out * isw + i;
                            let mut out_vecs = repeat_out_vecs!(out, out_idx_common, repeats);
                            repeat_mul_add!(inp_vec, krs, out_vecs, repeats);
                            repeat_write!(out, out_vecs, out_idx_common, repeats);
                        }
                    }
                    kernel.add(IC_NVEC * T::Vec::SIZE);
                }
            }
        } else {
            kernel.add(IC_NVEC * T::Vec::SIZE * (kw as usize) * ((o_end - oo) as usize));
        }
    }
}

pub(crate) struct PartialParams {
    pub(crate) arg1: [i64; 2],
    pub(crate) arg2: [i64; 2],
    pub(crate) arg3: [i64; 4],
    pub(crate) arg4: [i64; 3],
    pub(crate) arg5: [i64; 2],
    pub(crate) arg6: [i64; 3],
    pub(crate) arg7: [i64; 2],
    pub(crate) arg8: [i64; 2],
    pub(crate) arg9: [i64; 2],
    pub(crate) ic_remain: i64,
}

#[duplicate_item(
    template_function;
    [micro_kernel_5_1];
    [micro_kernel_4_1];
    [micro_kernel_3_1];
    [micro_kernel_2_1];
    [micro_kernel_1_1];
)]
pub(crate) fn template_function<T: CommonBounds>(
    params: PartialParams,
    mut out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: Cast<T>,
{
    let PartialParams {
        arg1: [oo, o_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
        ic_remain,
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    const IW_BLOCK: usize = OW_BLOCK;
    for n in 0..kh {
        let h_out = l * step_height + n * dh - ph_start;
        let h_in_range = h_out >= 0 && h_out < out_height;
        if h_in_range {
            for m in 0..kw {
                for o in oo..o_end {
                    for j in 0..ic_remain {
                        let kr = kernel[j];
                        let j = i + j;
                        for kk in 0..IW_BLOCK as i64 {
                            let w_out = (k + kk) * step_width + m * dw - pw_start;
                            if w_out >= 0 && w_out < out_width {
                                let out_idx = b * isb + h_out * ish + w_out * isw + j;
                                let inp_idx = b * osb + l * osh + (k + kk) * osw + o;
                                let inp = inp[inp_idx];
                                out[out_idx] = inp._mul_add(kr, out[out_idx]);
                            }
                        }
                    }
                    kernel.add(ic_remain as usize);
                }
            }
        } else {
            kernel.add(ic_remain as usize * (kw as usize) * ((o_end - oo) as usize));
        }
    }
}

pub(crate) fn full_oc_kernel_dispatch<T: CommonBounds>(
    oc: &mut usize, // output channels block size
    kb: &mut usize, // outwidth block size
) -> fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec)
where
    bool: Cast<T>,
{
    let kernels = [
        [
            micro_kernel_1x1,
            micro_kernel_2x1,
            micro_kernel_3x1,
            micro_kernel_4x1,
            micro_kernel_5x1,
        ],
        [
            micro_kernel_1x2,
            micro_kernel_2x2,
            micro_kernel_3x2,
            micro_kernel_4x2,
            micro_kernel_5x2,
        ],
        [
            micro_kernel_1x4,
            micro_kernel_2x4,
            micro_kernel_3x4,
            micro_kernel_4x4,
            micro_kernel_5x4,
        ],
        [
            micro_kernel_1x8,
            micro_kernel_2x8,
            micro_kernel_3x8,
            micro_kernel_4x8,
            micro_kernel_5x8,
        ],
    ];

    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;
    let map_oc = map_oc(*oc);
    if map_oc == 0 {
        *oc = 1;
    } else if map_oc == 1 {
        *oc = 2;
    } else if map_oc == 2 {
        *oc = 4;
    } else {
        *oc = 8;
    }

    kernels[map_oc][map_kb]
}

pub(crate) fn remain_ic_kernel_dispatch<T: CommonBounds>(
    kb: &mut usize, // outwidth block size
) -> fn(PartialParams, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec)
where
    bool: Cast<T>,
{
    let kernels = [
        micro_kernel_1_1,
        micro_kernel_2_1,
        micro_kernel_3_1,
        micro_kernel_4_1,
        micro_kernel_5_1,
    ];

    // println!("picked iconv2d_remain_microkernel_{} at {}", kb, map_kb(kb));
    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;

    kernels[map_kb]
}

fn map_kb(kb: usize) -> usize {
    if kb <= 1 {
        0
    } else if kb <= 2 {
        1
    } else if kb <= 3 {
        2
    } else if kb <= 4 {
        3
    } else {
        4
    }
}

fn map_oc(oc: usize) -> usize {
    if oc <= 1 {
        0
    } else if oc <= 2 {
        1
    } else if oc <= 4 {
        2
    } else {
        3
    }
}
