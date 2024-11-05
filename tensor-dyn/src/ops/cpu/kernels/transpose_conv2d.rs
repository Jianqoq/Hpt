use duplicate::duplicate_item;
use tensor_common::pointer::Pointer;
use tensor_macros::{
    conv2d_microkernel_declare_const,
    conv2d_microkernel_gen_inps,
    conv2d_microkernel_gen_kernels,
    conv2d_microkernel_gen_results,
    pwconv2d_microkernel_gen_pad_inps,
    transpose_conv2d_microkernel_gen_masks,
    transpose_conv2d_microkernel_gen_outs,
    transpose_conv2d_microkernel_gen_pad_results,
    transpose_conv2d_microkernel_gen_results,
};
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };
use tensor_traits::CommonBounds;
use tensor_types::traits::*;

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

/// This struct carries the micro kernel function and the corresponding info
#[derive(Clone, Copy)]
pub struct ConvPartialKernel<T: CommonBounds> {
    pub(crate) kernel: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec
    ),
    pub(crate) iw_block: usize,
}

/// This struct carries the micro kernel function and the corresponding info
#[derive(Clone, Copy)]
pub struct ConvKernel<T: CommonBounds> {
    pub(crate) kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T::Vec) -> T::Vec
    ),
}

impl<T: CommonBounds> ConvKernel<T> {
    pub(crate) fn new(
        kernel: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec)
    ) -> Self {
        Self {
            kernel,
        }
    }
}

impl<T: CommonBounds> ConvPartialKernel<T> {
    pub(crate) fn new(
        kernel: fn(
            PartialParams,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec
        ),
        iw_block: usize
    ) -> Self {
        Self { kernel, iw_block }
    }
}

macro_rules! repeat_inp {
    ($name:ident, $is3:expr, $step_width_isw:expr, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                T::Vec::splat($name[$is3 + $idx * $step_width_isw]),
            )*)
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
        $osw:ident,
        $img_width:ident,
        $pw_start:ident,
        $l_in_range:ident,
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            ($(
                {
                    let mask =
                    ($k + $idx) * $step_width + $m * $dw >= $pw_start &&
                    ($k + $idx) * $step_width + $m * $dw < $img_width + $pw_start;
                    let tmp_mask: T = mask.into_scalar();
                    let val = $name[($is3 + $idx * $step_width * $osw) * mask as i64];
                    T::Vec::splat(tmp_mask._mul(val))
                },
            )*)
        }
    };
}

macro_rules! repeat_pad_mask {
    (
        $k:ident,
        $step_width:ident,
        $m:ident,
        $dw:ident,
        $img_width:ident,
        $pw_start:ident,
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            ($(
                {
                    let mask =
                    ($k + $idx) * $step_width + $m * $dw >= $pw_start &&
                    ($k + $idx) * $step_width + $m * $dw < $img_width + $pw_start;
                    let tmp_mask: T = mask.into_scalar();
                    (tmp_mask, mask)
                },
            )*)
        }
    };
}

macro_rules! repeat_pad_outs {
    ($name:ident, $masks:ident, $is3:expr, $step_width:ident, $osw:ident, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                {
                    let val = $name[($is3 + $idx * $step_width * $osw) * ($masks.$idx.1 as i64)];
                    T::Vec::splat($masks.$idx.0._mul(val))
                },
            )*)
        }
    };
}

macro_rules! repeat_pw_pad_inp {
    (
        $name:ident,
        $is3:expr,
        $k:ident,
        $step_width:ident,
        $osw:ident,
        $img_width:ident,
        $pw_start:ident,
        $l_in_range:ident,
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            ($(
                {
                    let mask =
                    ($k + $idx) * $step_width >= $pw_start &&
                    ($k + $idx) * $step_width < $img_width + $pw_start;
                    let tmp_mask: T = mask.into_scalar();
                    let val = $name[($is3 + $idx * $step_width * $osw) * mask as i64];
                    T::Vec::splat(tmp_mask._mul(val))
                },
            )*)
        }
    };
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

macro_rules! repeat_results {
    (
        $results:ident,
        $inp:ident,
        $kernel:ident,
        [$vidx:literal, $($v:literal),*],
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            $(
                $results[$vidx][$idx] = $inp.$idx.mul_add($kernel.$vidx, $results[$vidx][$idx]);
            )*
            repeat_results!($results, $inp, $kernel, [$($v),*], [$($idx),*]);
        }
    };
    ($results:ident, $inp:ident, $kernel:ident, [$vidx:literal], [$($idx:expr),*]) => {
        paste::paste! {
            $(
                $results[$vidx][$idx] = $inp.$idx.mul_add($kernel.$vidx, $results[$vidx][$idx]);
            )*
        }
    };
}

macro_rules! repeat_transpose_results {
    (
        $results_vec:ident,
        $results:ident,
        $inp:ident,
        $kernel:ident,
        $is3:expr,
        $step_width:ident,
        $osw:ident,
        [$vidx:literal, $($v:literal),*],
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            $(
                $results[$is3 + $idx * ($step_width * $osw)] = $inp[$vidx][$idx].mul_add($kernel.$vidx, $results_vec.$idx).sum();
            )*
            repeat_transpose_results!($results_vec, $results, $inp, $kernel, $is3, $step_width, $osw, [$($v),*], [$($idx),*]);
        }
    };
    (
        $results_vec:ident,
        $results:ident,
        $inp:ident,
        $kernel:ident,
        $is3:expr,
        $step_width:ident,
        $osw:ident,
        [$vidx:literal],
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            $(
                $results[$is3 + $idx * ($step_width * $osw)] = $inp[$vidx][$idx].mul_add($kernel.$vidx, $results_vec.$idx).sum();
            )*
        }
    };
}

macro_rules! repeat_transpose_pad_results {
    (
        $results_vec:ident,
        $results:ident,
        $inp:ident,
        $kernel:ident,
        $is3:expr,
        $step_width:ident,
        $osw:ident,
        $masks:ident,
        [$vidx:literal, $($v:literal),*],
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            $(
                $results[($is3 + $idx * ($step_width * $osw)) * ($masks.$idx.1 as i64)] = $inp[$vidx][$idx].mul_add($kernel.$vidx, $results_vec.$idx).sum();
            )*
            repeat_transpose_pad_results!($results_vec, $results, $inp, $kernel, $is3, $step_width, $osw, $masks, [$($v),*], [$($idx),*]);
        }
    };
    (
        $results_vec:ident,
        $results:ident,
        $inp:ident,
        $kernel:ident,
        $is3:expr,
        $step_width:ident,
        $osw:ident,
        $masks:ident,
        [$vidx:literal],
        [$($idx:expr),*]
    ) => {
        paste::paste! {
            $(
                $results[($is3 + $idx * ($step_width * $osw)) * ($masks.$idx.1 as i64)] = $inp[$vidx][$idx].mul_add($kernel.$vidx, $results_vec.$idx).sum();
            )*
        }
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
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let Params {
        arg1: [jj, j_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let mut inps = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..OC_BLOCK {
            inps[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(
                        &inp[b * isb
                            + l * ish
                            + (k + kk) * isw
                            + i
                            + v as i64 * T::Vec::SIZE as i64] as *const _
                            as *const T,
                    )
                }; // prettier-ignore
        }
    }
    let is0 =
        b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    let m_must_in_range =
        k * step_width >= pw_start &&
        (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < out_width + pw_start;
    for n in 0..kh {
        let l_in_range =
            l * step_height + n * dh >= ph_start &&
            l * step_height + n * dh < out_height + ph_start;
        let is1 = is0 + n * dh * osh;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * osw;
                    for j in jj..j_end {
                        let is3 = is2 + j;
                        let outs = conv2d_microkernel_gen_inps!(
                            out,
                            is3,
                            step_width * osw,
                            template_function
                        );
                        let kernel_vecs = conv2d_microkernel_gen_kernels!(
                            kernel,
                            template_function
                        );
                        transpose_conv2d_microkernel_gen_results!(
                            out,
                            inps,
                            kernel_vecs,
                            template_function
                        );
                        kernel.add(OC_BLOCK * T::Vec::SIZE);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * osw;
                    for i in jj..j_end {
                        let is3 = is2 + i;
                        let masks = transpose_conv2d_microkernel_gen_masks!(
                            inp,
                            is3,
                            step_width * osw,
                            template_function
                        );
                        let outs = transpose_conv2d_microkernel_gen_outs!(
                            inp,
                            is3,
                            step_width * osw,
                            template_function
                        );
                        let kernel_vecs = conv2d_microkernel_gen_kernels!(
                            kernel,
                            template_function
                        );
                        transpose_conv2d_microkernel_gen_pad_results!(
                            out,
                            inps,
                            kernel_vecs,
                            template_function
                        );
                        kernel.add(OC_BLOCK * T::Vec::SIZE);
                    }
                }
            }
        } else {
            kernel.add(OC_BLOCK * T::Vec::SIZE * (kw as usize) * ((j_end - jj) as usize));
        }
    }
}

#[duplicate_item(
    template_function;
    [pw_micro_kernel_5x1];
    [pw_micro_kernel_4x1];
    [pw_micro_kernel_3x1];
    [pw_micro_kernel_2x1];
    [pw_micro_kernel_1x1];
    [pw_micro_kernel_5x2];
    [pw_micro_kernel_4x2];
    [pw_micro_kernel_3x2];
    [pw_micro_kernel_2x2];
    [pw_micro_kernel_1x2];
    [pw_micro_kernel_5x4];
    [pw_micro_kernel_4x4];
    [pw_micro_kernel_3x4];
    [pw_micro_kernel_2x4];
    [pw_micro_kernel_1x4];
    [pw_micro_kernel_5x8];
    [pw_micro_kernel_4x8];
    [pw_micro_kernel_3x8];
    [pw_micro_kernel_2x8];
    [pw_micro_kernel_1x8];
)]
fn template_function<T: CommonBounds>(
    params: Params,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let Params {
        arg1: [ii, i_end],
        arg2: [_, _],
        arg3: [b, l, k, j],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [_, _],
        arg9: [img_height, img_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..OC_BLOCK {
                ret[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(
                        &out[b * isb
                            + l * ish
                            + (k + kk) * isw
                            + j
                            + v as i64 * T::Vec::SIZE as i64] as *const _
                            as *const T,
                    )
                }; // prettier-ignore
            }
        }
        ret
    };
    let is0 =
        b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    let m_must_in_range =
        k * step_width >= pw_start && (k + (OW_BLOCK as i64)) * step_width < img_width + pw_start;

    let l_in_range = l * step_height >= ph_start && l * step_height < img_height + ph_start;
    if l_in_range {
        if m_must_in_range {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = conv2d_microkernel_gen_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let kernel_vecs = conv2d_microkernel_gen_kernels!(kernel, template_function);
                conv2d_microkernel_gen_results!(results, inp, kernel_vecs, template_function);
                kernel.add(OC_BLOCK * T::Vec::SIZE);
            }
        } else {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = pwconv2d_microkernel_gen_pad_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let kernel_vecs = conv2d_microkernel_gen_kernels!(kernel, template_function);
                conv2d_microkernel_gen_results!(results, inp, kernel_vecs, template_function);
                kernel.add(OC_BLOCK * T::Vec::SIZE);
            }
        }
    } else {
        kernel.add(OC_BLOCK * T::Vec::SIZE * ((i_end - ii) as usize));
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..OC_BLOCK {
            let out_vec = &mut out
                [b * isb + l * ish + (k + kk) * isw + j + (v * T::Vec::SIZE) as i64]
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
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let Params {
        arg1: [jj, j_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    // let mut results = if jj == 0 {
    //     [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK]
    // } else {
    //     let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
    //     for kk in 0..OW_BLOCK as i64 {
    //         for v in 0..OC_BLOCK {
    //             ret[v as usize][kk as usize] = unsafe {
    //                 T::Vec::from_ptr(
    //                     &out[b * isb
    //                         + l * ish
    //                         + (k + kk) * isw
    //                         + i
    //                         + v as i64 * T::Vec::SIZE as i64] as *const _
    //                         as *const T,
    //                 )
    //             }; // prettier-ignore
    //         }
    //     }
    //     ret
    // };
    // let is0 =
    //     b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    // let m_must_in_range =
    //     k * step_width >= pw_start &&
    //     (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < out_width + pw_start;
    // for n in 0..kh {
    //     let l_in_range =
    //         l * step_height + n * dh >= ph_start &&
    //         l * step_height + n * dh < out_height + ph_start;
    //     let is1 = is0 + n * dh * osh;
    //     if l_in_range {
    //         if m_must_in_range {
    //             for m in 0..kw {
    //                 let is2 = is1 + m * dw * osw;
    //                 for j in jj..j_end {
    //                     let is3 = is2 + j;
    //                     let inp = conv2d_microkernel_gen_inps!(
    //                         inp,
    //                         is3,
    //                         step_width * osw,
    //                         template_function
    //                     );
    //                     let kernel_vecs = conv2d_microkernel_gen_kernels!(
    //                         kernel,
    //                         template_function
    //                     );
    //                     conv2d_microkernel_gen_results!(
    //                         results,
    //                         inp,
    //                         kernel_vecs,
    //                         template_function
    //                     );
    //                     kernel.add(OC_BLOCK * T::Vec::SIZE);
    //                 }
    //             }
    //         } else {
    //             for m in 0..kw {
    //                 let is2 = is1 + m * dw * osw;
    //                 for i in jj..j_end {
    //                     let is3 = is2 + i;
    //                     let inp = transpose_conv2d_microkernel_gen_pad_inps!(
    //                         inp,
    //                         is3,
    //                         step_width * osw,
    //                         template_function
    //                     );
    //                     let kernel_vecs = conv2d_microkernel_gen_kernels!(
    //                         kernel,
    //                         template_function
    //                     );
    //                     conv2d_microkernel_gen_results!(
    //                         results,
    //                         inp,
    //                         kernel_vecs,
    //                         template_function
    //                     );
    //                     kernel.add(OC_BLOCK * T::Vec::SIZE);
    //                 }
    //             }
    //         }
    //     } else {
    //         kernel.add(OC_BLOCK * T::Vec::SIZE * (kw as usize) * ((j_end - jj) as usize));
    //     }
    // }
    // for kk in 0..OW_BLOCK as i64 {
    //     for v in 0..OC_BLOCK {
    //         let out_vec = &mut out
    //             [b * isb + l * ish + (k + kk) * isw + i + (v * T::Vec::SIZE) as i64]
    //             as *mut _ as *mut T::Vec; // prettier-ignore
    //         let bias_vec = unsafe {
    //             T::Vec::from_ptr(&bias[i + ((v * T::Vec::SIZE) as i64)] as *const _ as *const T)
    //         };
    //         unsafe {
    //             out_vec.write_unaligned(
    //                 activation(results[v as usize][kk as usize]._add(bias_vec))
    //             );
    //         }
    //     }
    // }
}

#[duplicate_item(
    template_function;
    [pw_bias_micro_kernel_5x1];
    [pw_bias_micro_kernel_4x1];
    [pw_bias_micro_kernel_3x1];
    [pw_bias_micro_kernel_2x1];
    [pw_bias_micro_kernel_1x1];
    [pw_bias_micro_kernel_5x2];
    [pw_bias_micro_kernel_4x2];
    [pw_bias_micro_kernel_3x2];
    [pw_bias_micro_kernel_2x2];
    [pw_bias_micro_kernel_1x2];
    [pw_bias_micro_kernel_5x4];
    [pw_bias_micro_kernel_4x4];
    [pw_bias_micro_kernel_3x4];
    [pw_bias_micro_kernel_2x4];
    [pw_bias_micro_kernel_1x4];
    [pw_bias_micro_kernel_5x8];
    [pw_bias_micro_kernel_4x8];
    [pw_bias_micro_kernel_3x8];
    [pw_bias_micro_kernel_2x8];
    [pw_bias_micro_kernel_1x8];
)]
fn template_function<T: CommonBounds>(
    params: Params,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let Params {
        arg1: [ii, i_end],
        arg2: [_, _],
        arg3: [b, l, k, j],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [_, _],
        arg9: [img_height, img_width],
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..OC_BLOCK {
                ret[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(
                        &out[b * isb
                            + l * ish
                            + (k + kk) * isw
                            + j
                            + v as i64 * T::Vec::SIZE as i64] as *const _
                            as *const T,
                    )
                }; // prettier-ignore
            }
        }
        ret
    };
    let is0 =
        b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    let m_must_in_range =
        k * step_width >= pw_start && (k + (OW_BLOCK as i64)) * step_width < img_width + pw_start;
    let l_in_range = l * step_height >= ph_start && l * step_height < img_height + ph_start;
    if l_in_range {
        if m_must_in_range {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = conv2d_microkernel_gen_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let kernel_vecs = conv2d_microkernel_gen_kernels!(kernel, template_function);
                conv2d_microkernel_gen_results!(results, inp, kernel_vecs, template_function);
                kernel.add(OC_BLOCK * T::Vec::SIZE);
            }
        } else {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = pwconv2d_microkernel_gen_pad_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let kernel_vecs = conv2d_microkernel_gen_kernels!(kernel, template_function);
                conv2d_microkernel_gen_results!(results, inp, kernel_vecs, template_function);
                kernel.add(OC_BLOCK * T::Vec::SIZE);
            }
        }
    } else {
        kernel.add(OC_BLOCK * T::Vec::SIZE * ((i_end - ii) as usize));
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..OC_BLOCK {
            let out_vec = &mut out
                [b * isb + l * ish + (k + kk) * isw + j + (v * T::Vec::SIZE) as i64]
                as *mut _ as *mut T::Vec; // prettier-ignore
            let bias_vec = unsafe {
                T::Vec::from_ptr(&bias[j + ((v * T::Vec::SIZE) as i64)] as *const _ as *const T)
            };
            unsafe {
                out_vec.write_unaligned(
                    activation(results[v as usize][kk as usize]._add(bias_vec))
                );
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
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let PartialParams {
        arg1: [jj, j_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
        ic_remain,
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let mut inps = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..ic_remain {
            inps[0][kk as usize][v as usize] = inp[b * isb + l * ish + (k + kk) * isw + i + v];
        }
    }
    let is0 =
        b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    let m_must_in_range =
        k * step_width >= pw_start &&
        (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < out_width + pw_start;
    for n in 0..kh {
        let l_in_range =
            l * step_height + n * dh >= ph_start &&
            l * step_height + n * dh < out_height + ph_start;
        let is1 = is0 + n * dh * osh;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * osw;
                    for i in jj..j_end {
                        let is3 = is2 + i;
                        let outs = conv2d_microkernel_gen_inps!(
                            out,
                            is3,
                            step_width * osw,
                            template_function
                        );
                        let mut kernel0 = (T::Vec::splat(T::ZERO),);
                        for v in 0..ic_remain {
                            kernel0.0[v as usize] = kernel[v as usize];
                        }
                        transpose_conv2d_microkernel_gen_results!(
                            out,
                            inps,
                            kernel0,
                            template_function
                        );
                        // conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                        kernel.add(ic_remain as usize);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * osw;
                    for i in jj..j_end {
                        let is3 = is2 + i;
                        let masks = transpose_conv2d_microkernel_gen_masks!(
                            inp,
                            is3,
                            step_width * osw,
                            template_function
                        );
                        let outs = transpose_conv2d_microkernel_gen_outs!(
                            inp,
                            is3,
                            step_width * osw,
                            template_function
                        );
                        let mut kernel0 = (T::Vec::splat(T::ZERO),);
                        for v in 0..ic_remain {
                            kernel0.0[v as usize] = kernel[v as usize];
                        }
                        transpose_conv2d_microkernel_gen_results!(
                            out,
                            inps,
                            kernel0,
                            template_function
                        );
                        kernel.add(ic_remain as usize);
                    }
                }
            }
        } else {
            kernel.add((ic_remain as usize) * (kw as usize) * ((j_end - jj) as usize));
        }
    }
}

#[duplicate_item(
    template_function;
    [pw_micro_kernel_5_1];
    [pw_micro_kernel_4_1];
    [pw_micro_kernel_3_1];
    [pw_micro_kernel_2_1];
    [pw_micro_kernel_1_1];
)]
#[inline]
fn template_function<T: CommonBounds>(
    params: PartialParams,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let PartialParams {
        arg1: [ii, i_end],
        arg2: [_, _],
        arg3: [b, l, k, j],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        arg7: [ph_start, pw_start],
        arg8: [_, _],
        arg9: [img_height, img_width],
        ic_remain,
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..ic_remain {
                ret[0][kk as usize][v as usize] = out[b * isb + l * ish + (k + kk) * isw + j + v];
            }
        }
        ret
    };
    let is0 =
        b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    let m_must_in_range =
        k * step_width >= pw_start && (k + (OW_BLOCK as i64)) * step_width < img_width + pw_start;

    let l_in_range = l * step_height >= ph_start && l * step_height < img_height + ph_start;
    if l_in_range {
        if m_must_in_range {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = conv2d_microkernel_gen_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let mut kernel0 = (T::Vec::splat(T::ZERO),);
                for v in 0..ic_remain {
                    kernel0.0[v as usize] = kernel[v as usize];
                }
                conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                kernel.add(ic_remain as usize);
            }
        } else {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = pwconv2d_microkernel_gen_pad_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let mut kernel0 = (T::Vec::splat(T::ZERO),);
                for v in 0..ic_remain {
                    kernel0.0[v as usize] = kernel[v as usize];
                }
                conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                kernel.add(ic_remain as usize);
            }
        }
    } else {
        kernel.add((ic_remain as usize) * ((i_end - ii) as usize));
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..ic_remain {
            let res = activation(results[0][kk as usize]);
            out[b * isb + l * ish + (k + kk) * isw + j + v] = res[v as usize];
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
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let PartialParams {
        arg1: [jj, j_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
        ic_remain,
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    todo!()
    // let mut results = if jj == 0 {
    //     [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1]
    // } else {
    //     let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
    //     for kk in 0..OW_BLOCK as i64 {
    //         for v in 0..ic_remain {
    //             ret[0][kk as usize][v as usize] = out[b * isb + l * ish + (k + kk) * isw + i + v];
    //         }
    //     }
    //     ret
    // };
    // let is0 =
    //     b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    // let m_must_in_range =
    //     k * step_width >= pw_start &&
    //     (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < out_width + pw_start;
    // for n in 0..kh {
    //     let is1 = is0 + n * dh * osh;
    //     let l_in_range =
    //         l * step_height + n * dh >= ph_start &&
    //         l * step_height + n * dh < out_height + ph_start;
    //     if l_in_range {
    //         if m_must_in_range {
    //             for m in 0..kw {
    //                 let is2 = is1 + m * dw * osw;
    //                 for i in jj..j_end {
    //                     let is3 = is2 + i;
    //                     let inp = conv2d_microkernel_gen_inps!(
    //                         inp,
    //                         is3,
    //                         step_width * osw,
    //                         template_function
    //                     );
    //                     let mut kernel0 = (T::Vec::splat(T::ZERO),);
    //                     for v in 0..ic_remain {
    //                         kernel0.0[v as usize] = kernel[v as usize];
    //                     }
    //                     conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
    //                     kernel.add(ic_remain as usize);
    //                 }
    //             }
    //         } else {
    //             for m in 0..kw {
    //                 let is2 = is1 + m * dw * osw;
    //                 for i in jj..j_end {
    //                     let is3 = is2 + i;
    //                     let inp = transpose_conv2d_microkernel_gen_pad_inps!(
    //                         inp,
    //                         is3,
    //                         step_width * osw,
    //                         template_function
    //                     );
    //                     let mut kernel0 = (T::Vec::splat(T::ZERO),);
    //                     for v in 0..ic_remain {
    //                         kernel0.0[v as usize] = kernel[v as usize];
    //                     }
    //                     conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
    //                     kernel.add(ic_remain as usize);
    //                 }
    //             }
    //         }
    //     } else {
    //         kernel.add((ic_remain as usize) * (kw as usize) * ((j_end - jj) as usize));
    //     }
    // }
    // for kk in 0..OW_BLOCK as i64 {
    //     for v in 0..ic_remain as usize {
    //         results[0][kk as usize][v] = results[0][kk as usize][v]._add(bias[i + (v as i64)]);
    //     }
    //     let res = activation(results[0][kk as usize]);
    //     for v in 0..ic_remain as usize {
    //         out[b * isb + l * ish + (k + kk) * isw + i + (v as i64)] = res[v];
    //     }
    // }
}

#[duplicate_item(
    template_function;
    [pw_bias_micro_kernel_5_1];
    [pw_bias_micro_kernel_4_1];
    [pw_bias_micro_kernel_3_1];
    [pw_bias_micro_kernel_2_1];
    [pw_bias_micro_kernel_1_1];
)]
#[inline]
fn template_function<T: CommonBounds>(
    params: PartialParams,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec
)
    where bool: IntoScalar<T>
{
    let PartialParams {
        arg1: [ii, i_end],
        arg2: [_, _],
        arg3: [b, l, k, j],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        arg7: [ph_start, pw_start],
        arg8: [_, _],
        arg9: [img_height, img_width],
        ic_remain,
    } = params;
    conv2d_microkernel_declare_const!(template_function);
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..ic_remain {
                ret[0][kk as usize][v as usize] = out[b * isb + l * ish + (k + kk) * isw + j + v];
            }
        }
        ret
    };
    let is0 =
        b * osb + l * step_height * osh + k * step_width * osw - pw_start * osw - ph_start * osh;
    let m_must_in_range =
        k * step_width >= pw_start && (k + (OW_BLOCK as i64)) * step_width < img_width + pw_start;

    let l_in_range = l * step_height >= ph_start && l * step_height < img_height + ph_start;
    if l_in_range {
        if m_must_in_range {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = conv2d_microkernel_gen_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let mut kernel0 = (T::Vec::splat(T::ZERO),);
                for v in 0..ic_remain {
                    kernel0.0[v as usize] = kernel[v as usize];
                }
                conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                kernel.add(ic_remain as usize);
            }
        } else {
            for i in ii..i_end {
                let is3 = is0 + i;
                let inp = pwconv2d_microkernel_gen_pad_inps!(
                    inp,
                    is3,
                    step_width * osw,
                    template_function
                );
                let mut kernel0 = (T::Vec::splat(T::ZERO),);
                for v in 0..ic_remain {
                    kernel0.0[v as usize] = kernel[v as usize];
                }
                conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                kernel.add(ic_remain as usize);
            }
        }
    } else {
        kernel.add((ic_remain as usize) * ((i_end - ii) as usize));
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..ic_remain as usize {
            results[0][kk as usize][v] = results[0][kk as usize][v]._add(bias[j + (v as i64)]);
        }
        let res = activation(results[0][kk as usize]);
        for v in 0..ic_remain as usize {
            out[b * isb + l * ish + (k + kk) * isw + j + (v as i64)] = res[v];
        }
    }
}

pub(crate) fn tconv2d_full_ic_dispatch<T: CommonBounds>(
    [kh, kw]: [i64; 2],
    oc: &mut usize, // output channels block size
    kb: &mut usize // outwidth block size
) -> Option<ConvKernel<T>>
    where bool: IntoScalar<T>
{
    let kernels: [
        [fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec); 5];
        4
    ] = if kh == 1 && kw == 1 {
        [
            [
                pw_micro_kernel_1x1,
                pw_micro_kernel_2x1,
                pw_micro_kernel_3x1,
                pw_micro_kernel_4x1,
                pw_micro_kernel_5x1,
            ],
            [
                pw_micro_kernel_1x2,
                pw_micro_kernel_2x2,
                pw_micro_kernel_3x2,
                pw_micro_kernel_4x2,
                pw_micro_kernel_5x2,
            ],
            [
                pw_micro_kernel_1x4,
                pw_micro_kernel_2x4,
                pw_micro_kernel_3x4,
                pw_micro_kernel_4x4,
                pw_micro_kernel_5x4,
            ],
            [
                pw_micro_kernel_1x8,
                pw_micro_kernel_2x8,
                pw_micro_kernel_3x8,
                pw_micro_kernel_4x8,
                pw_micro_kernel_5x8,
            ],
        ]
    } else {
        [
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
        ]
    };

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

    let kernel_fn = kernels
        .get(map_oc)
        .map(|x| x.get(map_kb))
        .flatten();

    // println!("picked iconv2d_microkernel_{}x{} at {}{}", kb, oc, map_oc, map_kb);

    kernel_fn.cloned().map(|kernel| ConvKernel::new(kernel))
}

pub(crate) fn conv2d_full_oc_bias_kernel_dispatch<T: CommonBounds>(
    [kh, kw]: [i64; 2],
    oc: &mut usize, // output channels block size
    kb: &mut usize // outwidth block size
) -> Option<
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, &Pointer<T>, fn(T::Vec) -> T::Vec)
    >
    where bool: IntoScalar<T>
{
    let kernels: [
        [
            fn(
                Params,
                &mut Pointer<T>,
                &mut Pointer<T>,
                &Pointer<T>,
                &Pointer<T>,
                fn(T::Vec) -> T::Vec
            );
            5
        ];
        4
    ] = if kh == 1 && kw == 1 {
        [
            [
                pw_bias_micro_kernel_1x1,
                pw_bias_micro_kernel_2x1,
                pw_bias_micro_kernel_3x1,
                pw_bias_micro_kernel_4x1,
                pw_bias_micro_kernel_5x1,
            ],
            [
                pw_bias_micro_kernel_1x2,
                pw_bias_micro_kernel_2x2,
                pw_bias_micro_kernel_3x2,
                pw_bias_micro_kernel_4x2,
                pw_bias_micro_kernel_5x2,
            ],
            [
                pw_bias_micro_kernel_1x4,
                pw_bias_micro_kernel_2x4,
                pw_bias_micro_kernel_3x4,
                pw_bias_micro_kernel_4x4,
                pw_bias_micro_kernel_5x4,
            ],
            [
                pw_bias_micro_kernel_1x8,
                pw_bias_micro_kernel_2x8,
                pw_bias_micro_kernel_3x8,
                pw_bias_micro_kernel_4x8,
                pw_bias_micro_kernel_5x8,
            ],
        ]
    } else {
        [
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
        ]
    };

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

    let kernel_fn = kernels
        .get(map_oc)
        .map(|x| x.get(map_kb))
        .flatten();

    kernel_fn.cloned()
}

pub(crate) fn remain_oc_kernel_dispatch<T: CommonBounds>(
    [kh, kw]: [i64; 2],
    kb: &mut usize // outwidth block size
) -> Option<ConvPartialKernel<T>>
    where bool: IntoScalar<T>
{
    let kernels: [ConvPartialKernel<T>; 5] = if kh == 1 && kw == 1 {
        [
            ConvPartialKernel::new(pw_micro_kernel_1_1, 1),
            ConvPartialKernel::new(pw_micro_kernel_2_1, 2),
            ConvPartialKernel::new(pw_micro_kernel_3_1, 3),
            ConvPartialKernel::new(pw_micro_kernel_4_1, 4),
            ConvPartialKernel::new(pw_micro_kernel_5_1, 5),
        ]
    } else {
        [
            ConvPartialKernel::new(micro_kernel_1_1, 1),
            ConvPartialKernel::new(micro_kernel_2_1, 2),
            ConvPartialKernel::new(micro_kernel_3_1, 3),
            ConvPartialKernel::new(micro_kernel_4_1, 4),
            ConvPartialKernel::new(micro_kernel_5_1, 5),
        ]
    };

    // println!("picked iconv2d_remain_microkernel_{} at {}", kb, map_kb(kb));
    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;

    let kernel_fn = kernels.get(map_kb);

    kernel_fn.cloned()
}

pub(crate) fn bias_remain_oc_kernel_dispatch<T: CommonBounds>(
    [kh, kw]: [i64; 2],
    kb: &mut usize // outwidth block size
) -> Option<
        fn(
            PartialParams,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec
        )
    >
    where bool: IntoScalar<T>
{
    let kernels: [
        fn(
            PartialParams,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T::Vec) -> T::Vec
        );
        5
    ] = if kh == 1 && kw == 1 {
        [
            pw_bias_micro_kernel_1_1,
            pw_bias_micro_kernel_2_1,
            pw_bias_micro_kernel_3_1,
            pw_bias_micro_kernel_4_1,
            pw_bias_micro_kernel_5_1,
        ]
    } else {
        [
            bias_micro_kernel_1_1,
            bias_micro_kernel_2_1,
            bias_micro_kernel_3_1,
            bias_micro_kernel_4_1,
            bias_micro_kernel_5_1,
        ]
    };

    // println!("picked iconv2d_remain_microkernel_{} at {}", kb, map_kb(kb));
    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;

    let kernel_fn = kernels.get(map_kb);

    kernel_fn.cloned()
}

fn map_kb(kb: usize) -> usize {
    if kb <= 1 { 0 } else if kb <= 2 { 1 } else if kb <= 3 { 2 } else if kb <= 4 { 3 } else { 4 }
}

fn map_oc(oc: usize) -> usize {
    if oc <= 1 { 0 } else if oc <= 2 { 1 } else if oc <= 4 { 2 } else { 3 }
}
