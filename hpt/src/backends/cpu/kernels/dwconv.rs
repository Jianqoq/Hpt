use duplicate::duplicate_item;
use hpt_common::utils::pointer::Pointer;
use hpt_macros::{
    conv2d_microkernel_declare_const, conv2d_microkernel_gen_inps,
    dwconv2d_microkernel_gen_pad_inps, dwconv2d_microkernel_gen_results,
};
use hpt_traits::tensor::CommonBounds;
use hpt_types::traits::*;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};

pub(crate) struct Params {
    pub(crate) arg1: i64,
    pub(crate) arg2: [i64; 2],
    pub(crate) arg3: [i64; 3],
    pub(crate) arg4: [i64; 3],
    pub(crate) arg5: [i64; 2],
    pub(crate) arg6: [i64; 3],
    pub(crate) pads: [i64; 2],
    pub(crate) arg8: [i64; 2],
    pub(crate) arg9: [i64; 2],
}

pub(crate) struct PartialParams {
    pub(crate) arg1: i64,
    pub(crate) arg2: [i64; 2],
    pub(crate) arg3: [i64; 3],
    pub(crate) arg4: [i64; 3],
    pub(crate) arg5: [i64; 2],
    pub(crate) arg6: [i64; 3],
    pub(crate) arg7: [i64; 2],
    pub(crate) arg8: [i64; 2],
    pub(crate) arg9: [i64; 2],
    pub(crate) oc_remain: i64,
}

/// This struct carries the micro kernel function and the corresponding info
#[derive(Clone, Copy)]
pub struct ConvPartialKernel<T: CommonBounds> {
    pub(crate) kernel:
        fn(PartialParams, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
    pub(crate) ow_block: usize,
}

/// This struct carries the micro kernel function and the corresponding info
#[derive(Clone, Copy)]
pub struct ConvKernel<T: CommonBounds> {
    pub(crate) kernel:
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
}

impl<T: CommonBounds> ConvKernel<T> {
    pub(crate) fn new(
        kernel: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
    ) -> Self {
        Self { kernel }
    }
}

impl<T: CommonBounds> ConvPartialKernel<T> {
    pub(crate) fn new(
        kernel: fn(
            PartialParams,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec,
        ),
        ow_block: usize,
    ) -> Self {
        Self { kernel, ow_block }
    }
}

macro_rules! repeat_inp {
    ($name:ident, $is3:expr, $step_width_isw:expr, [$($idx:expr),*]) => {
        paste::paste! {
            [$(
                unsafe {
                    T::Vec::from_ptr(&$name[$is3 + $idx * $step_width_isw] as *const _ as *const T)
                },
            )*]
        }
    };
}

macro_rules! repeat_pad_inp {
    (
        $name:ident,
        $is3:expr,
        $k:ident,
        $step_width:ident,
        $m:ident,
        $dw:ident,
        $isw:ident,
        $img_width:ident,
        $pw_start:ident,
        $l_in_range:ident,
        $ic_remain: ident,
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            [$(
                {
                    let mut vec = T::Vec::splat(T::ZERO);
                    for i in 0..$ic_remain {
                        let inrange = ($k + $idx) * $step_width + $m * $dw >= $pw_start &&
                        ($k + $idx) * $step_width + $m * $dw < $img_width + $pw_start;
                        if inrange {
                            let val = $name[$is3 + $idx * $step_width * $isw + i as i64];
                            vec[i as usize] = val;
                        }
                    }
                    vec
                },
            )*]
        }
    };
}

macro_rules! repeat_results {
    ($results:ident, $inp:ident, $kernel:ident, $i:ident, [$($idx:expr),*]) => {
        $(
            $results[$i as usize][$idx] = $inp[$idx].mul_add($kernel, $results[$i as usize][$idx]);
        )*
    };
}

#[duplicate_item(
    template_function;
    [micro_kernel_5x1];
    [micro_kernel_4x1];
    [micro_kernel_3x1];
    [micro_kernel_2x1];
    [micro_kernel_1x1];
    [micro_kernel_5x2];
    [micro_kernel_4x2];
    [micro_kernel_3x2];
    [micro_kernel_2x2];
    [micro_kernel_1x2];
    [micro_kernel_5x4];
    [micro_kernel_4x4];
    [micro_kernel_3x4];
    [micro_kernel_2x4];
    [micro_kernel_1x4];
    [micro_kernel_5x8];
    [micro_kernel_4x8];
    [micro_kernel_3x8];
    [micro_kernel_2x8];
    [micro_kernel_1x8];
)]
fn template_function<T: CommonBounds>(
    params: Params,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: Cast<T>,
{
    let Params {
        arg1: ii,
        arg2: [kh, kw],
        arg3: [b, l, k],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    const IC_BLOCK: usize = OC_BLOCK;
    let ic_remain = T::Vec::SIZE;
    let mut results = [[T::Vec::splat(T::ZERO); OW_BLOCK]; IC_BLOCK];
    let is0 =
        b * isb + l * step_height * ish + k * step_width * isw - pw_start * isw - ph_start * ish
            + ii;
    let m_must_in_range = k * step_width >= pw_start
        && (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < img_width + pw_start;
    for n in 0..kh {
        let l_in_range = l * step_height + n * dh >= ph_start
            && l * step_height + n * dh < img_height + ph_start;
        let is1 = is0 + n * dh * ish;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..IC_BLOCK as i64 {
                        let is3 = is2 + i * (T::Vec::SIZE as i64);
                        let inp = conv2d_microkernel_gen_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let kernel0 = unsafe { T::Vec::from_ptr(&kernel[0usize]) };
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(T::Vec::SIZE);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..IC_BLOCK as i64 {
                        let is3 = is2 + i * (T::Vec::SIZE as i64);
                        let inp = dwconv2d_microkernel_gen_pad_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let kernel0 = unsafe { T::Vec::from_ptr(&kernel[0usize]) };
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(T::Vec::SIZE);
                    }
                }
            }
        } else {
            kernel.add(T::Vec::SIZE * (kw as usize) * IC_BLOCK);
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..IC_BLOCK {
            let out_vec = &mut out
                [b * osb + l * osh + (k + kk) * osw + (v * T::Vec::SIZE) as i64 + ii]
                as *mut _ as *mut T::Vec; // prettier-ignore
            unsafe {
                out_vec.write_unaligned(activation(results[v as usize][kk as usize]));
            }
        }
    }
}

#[duplicate_item(
    template_function;
    [bias_micro_kernel_5x1];
    [bias_micro_kernel_4x1];
    [bias_micro_kernel_3x1];
    [bias_micro_kernel_2x1];
    [bias_micro_kernel_1x1];
    [bias_micro_kernel_5x2];
    [bias_micro_kernel_4x2];
    [bias_micro_kernel_3x2];
    [bias_micro_kernel_2x2];
    [bias_micro_kernel_1x2];
    [bias_micro_kernel_5x4];
    [bias_micro_kernel_4x4];
    [bias_micro_kernel_3x4];
    [bias_micro_kernel_2x4];
    [bias_micro_kernel_1x4];
    [bias_micro_kernel_5x8];
    [bias_micro_kernel_4x8];
    [bias_micro_kernel_3x8];
    [bias_micro_kernel_2x8];
    [bias_micro_kernel_1x8];
)]
fn template_function<T: CommonBounds>(
    params: Params,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: Cast<T>,
{
    let Params {
        arg1: ii,
        arg2: [kh, kw],
        arg3: [b, l, k],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let ic_remain = T::Vec::SIZE;
    let mut results = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
    let is0 =
        b * isb + l * step_height * ish + k * step_width * isw - pw_start * isw - ph_start * ish
            + ii;
    let m_must_in_range = k * step_width >= pw_start
        && (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < img_width + pw_start;
    for n in 0..kh {
        let l_in_range = l * step_height + n * dh >= ph_start
            && l * step_height + n * dh < img_height + ph_start;
        let is1 = is0 + n * dh * ish;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..OC_BLOCK as i64 {
                        let is3 = is2 + i * (T::Vec::SIZE as i64);
                        let inp = conv2d_microkernel_gen_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let kernel0 = unsafe { T::Vec::from_ptr(&kernel[0usize]) };
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(T::Vec::SIZE);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..OC_BLOCK as i64 {
                        let is3 = is2 + i * (T::Vec::SIZE as i64);
                        let inp = dwconv2d_microkernel_gen_pad_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let kernel0 = unsafe { T::Vec::from_ptr(&kernel[0usize]) };
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(T::Vec::SIZE);
                    }
                }
            }
        } else {
            kernel.add(OC_BLOCK * T::Vec::SIZE * (kw as usize));
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..OC_BLOCK {
            let out_vec = &mut out
                [b * osb + l * osh + (k + kk) * osw + (v * T::Vec::SIZE) as i64 + ii]
                as *mut _ as *mut T::Vec; // prettier-ignore
            let bias_vec = unsafe {
                T::Vec::from_ptr(&bias[ii + (v * T::Vec::SIZE) as i64] as *const _ as *const T)
            };
            unsafe {
                out_vec
                    .write_unaligned(activation(results[v as usize][kk as usize]._add(bias_vec)));
            }
        }
    }
}

#[duplicate_item(
    template_function;
    [micro_kernel_5_1];
    [micro_kernel_4_1];
    [micro_kernel_3_1];
    [micro_kernel_2_1];
    [micro_kernel_1_1];
)]
#[inline]
fn template_function<T: CommonBounds>(
    params: PartialParams,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: Cast<T>,
{
    let PartialParams {
        arg1: ii,
        arg2: [kh, kw],
        arg3: [b, l, k],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
        oc_remain: ic_remain,
    } = params;
    assert!(ic_remain < T::Vec::SIZE as i64);
    conv2d_microkernel_declare_const!(template_function);
    let mut results = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
    let is0 =
        b * isb + l * step_height * ish + k * step_width * isw - pw_start * isw - ph_start * ish
            + ii;
    let m_must_in_range = k * step_width >= pw_start
        && (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < img_width + pw_start;
    for n in 0..kh {
        let l_in_range = l * step_height + n * dh >= ph_start
            && l * step_height + n * dh < img_height + ph_start;
        let is1 = is0 + n * dh * ish;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..1 as i64 {
                        let is3 = is2;
                        let inp = conv2d_microkernel_gen_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let mut kernel0 = T::Vec::splat(T::ZERO);
                        for v in 0..ic_remain {
                            kernel0[v as usize] = kernel[v as usize];
                        }
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(ic_remain as usize);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..1 as i64 {
                        let is3 = is2;
                        let inp = dwconv2d_microkernel_gen_pad_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let mut kernel0 = T::Vec::splat(T::ZERO);
                        for v in 0..ic_remain {
                            kernel0[v as usize] = kernel[v as usize];
                        }
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(ic_remain as usize);
                    }
                }
            }
        } else {
            kernel.add((ic_remain as usize) * (kw as usize));
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        let res = activation(results[0][kk as usize]);
        for v in 0..ic_remain {
            out[b * osb + l * osh + (k + kk) * osw + v + ii] = res[v as usize];
        }
    }
}

#[duplicate_item(
    template_function;
    [bias_micro_kernel_5_1];
    [bias_micro_kernel_4_1];
    [bias_micro_kernel_3_1];
    [bias_micro_kernel_2_1];
    [bias_micro_kernel_1_1];
)]
#[inline]
fn template_function<T: CommonBounds>(
    params: PartialParams,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: Cast<T>,
{
    let PartialParams {
        arg1: ii,
        arg2: [kh, kw],
        arg3: [b, l, k],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
        oc_remain: ic_remain,
    } = params;
    assert!(ic_remain < T::Vec::SIZE as i64);
    conv2d_microkernel_declare_const!(template_function);
    let mut results = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
    let is0 =
        b * isb + l * step_height * ish + k * step_width * isw - pw_start * isw - ph_start * ish
            + ii;
    let m_must_in_range = k * step_width >= pw_start
        && (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < img_width + pw_start;
    for n in 0..kh {
        let is1 = is0 + n * dh * ish;
        let l_in_range = l * step_height + n * dh >= ph_start
            && l * step_height + n * dh < img_height + ph_start;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..1 {
                        let is3 = is2 + i;
                        let inp = conv2d_microkernel_gen_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let mut kernel0 = T::Vec::splat(T::ZERO);
                        for v in 0..ic_remain {
                            kernel0[v as usize] = kernel[v as usize];
                        }
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(ic_remain as usize);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in 0..1 {
                        let is3 = is2 + i;
                        let inp = dwconv2d_microkernel_gen_pad_inps!(
                            inp,
                            is3,
                            step_width * isw,
                            template_function
                        );
                        let mut kernel0 = T::Vec::splat(T::ZERO);
                        for v in 0..ic_remain {
                            kernel0[v as usize] = kernel[v as usize];
                        }
                        dwconv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(ic_remain as usize);
                    }
                }
            }
        } else {
            kernel.add((ic_remain as usize) * (kw as usize));
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..ic_remain as usize {
            results[0][kk as usize][v] = results[0][kk as usize][v]._add(bias[ii + v as i64]);
        }
        let res = activation(results[0][kk as usize]);
        for v in 0..ic_remain as usize {
            out[b * osb + l * osh + (k + kk) * osw + (v as i64) + ii] = res[v];
        }
    }
}

pub(crate) fn conv2d_full_oc_kernel_dispatch<T: CommonBounds>(
    ic: &mut usize, // output channels block size
    kb: &mut usize, // outwidth block size
) -> Option<ConvKernel<T>>
where
    bool: Cast<T>,
{
    let kernels: [[fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec);
        5]; 4] = [
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
    let map_oc = map_oc(*ic);
    if map_oc == 0 {
        *ic = 1;
    } else if map_oc == 1 {
        *ic = 2;
    } else if map_oc == 2 {
        *ic = 4;
    } else {
        *ic = 8;
    }

    let kernel_fn = kernels.get(map_oc).map(|x| x.get(map_kb)).flatten();

    // println!("picked iconv2d_microkernel_{}x{} at {}{}", kb, oc, map_oc, map_kb);

    kernel_fn.cloned().map(|kernel| ConvKernel::new(kernel))
}

pub(crate) fn conv2d_full_oc_bias_kernel_dispatch<T: CommonBounds>(
    ic: &mut usize, // output channels block size
    kb: &mut usize, // outwidth block size
) -> Option<
    fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec),
>
where
    bool: Cast<T>,
{
    let kernels: [[fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ); 5]; 4] = [
        [
            bias_micro_kernel_1x1,
            bias_micro_kernel_2x1,
            bias_micro_kernel_3x1,
            bias_micro_kernel_4x1,
            bias_micro_kernel_5x1,
        ],
        [
            bias_micro_kernel_1x2,
            bias_micro_kernel_2x2,
            bias_micro_kernel_3x2,
            bias_micro_kernel_4x2,
            bias_micro_kernel_5x2,
        ],
        [
            bias_micro_kernel_1x4,
            bias_micro_kernel_2x4,
            bias_micro_kernel_3x4,
            bias_micro_kernel_4x4,
            bias_micro_kernel_5x4,
        ],
        [
            bias_micro_kernel_1x8,
            bias_micro_kernel_2x8,
            bias_micro_kernel_3x8,
            bias_micro_kernel_4x8,
            bias_micro_kernel_5x8,
        ],
    ];

    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;
    let map_ic = map_oc(*ic);
    if map_ic == 0 {
        *ic = 1;
    } else if map_ic == 1 {
        *ic = 2;
    } else if map_ic == 2 {
        *ic = 4;
    } else {
        *ic = 8;
    }

    let kernel_fn = kernels.get(map_ic).map(|x| x.get(map_kb)).flatten();

    kernel_fn.cloned()
}

pub(crate) fn remain_oc_kernel_dispatch<T: CommonBounds>(
    kb: &mut usize, // outwidth block size
) -> Option<ConvPartialKernel<T>>
where
    bool: Cast<T>,
{
    let kernels: [ConvPartialKernel<T>; 5] = [
        ConvPartialKernel::new(micro_kernel_1_1, 1),
        ConvPartialKernel::new(micro_kernel_2_1, 2),
        ConvPartialKernel::new(micro_kernel_3_1, 3),
        ConvPartialKernel::new(micro_kernel_4_1, 4),
        ConvPartialKernel::new(micro_kernel_5_1, 5),
    ];

    // println!("picked iconv2d_remain_microkernel_{} at {}", kb, map_kb(kb));
    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;

    let kernel_fn = kernels.get(map_kb);

    kernel_fn.cloned()
}

pub(crate) fn bias_remain_oc_kernel_dispatch<T: CommonBounds>(
    kb: &mut usize, // outwidth block size
) -> Option<
    fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ),
>
where
    bool: Cast<T>,
{
    let kernels: [fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec,
    ); 5] = [
        bias_micro_kernel_1_1,
        bias_micro_kernel_2_1,
        bias_micro_kernel_3_1,
        bias_micro_kernel_4_1,
        bias_micro_kernel_5_1,
    ];

    // println!("picked iconv2d_remain_microkernel_{} at {}", kb, map_kb(kb));
    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;

    let kernel_fn = kernels.get(map_kb);

    kernel_fn.cloned()
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
