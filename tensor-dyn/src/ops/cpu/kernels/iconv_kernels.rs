#![allow(unused)]
use duplicate::duplicate_item;
use tensor_common::pointer::Pointer;
use tensor_macros::{
    conv2d_microkernel_declare_const,
    conv2d_microkernel_gen_inps,
    conv2d_microkernel_gen_kernels,
    conv2d_microkernel_gen_results,
};
use tensor_traits::CommonBounds;
use tensor_types::traits::*;

macro_rules! repeat_inp {
    ($name:ident, $is3:expr, $step_width_m:expr, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                T::Vec::splat($name[$is3 + $idx * $step_width_m]),
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
    [ii, i_end]: [i64; 2],
    [kh, kw]: [i64; 2],
    [b, l, k, j]: [i64; 4],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    kernel: &mut Pointer<T>
) {
    conv2d_microkernel_declare_const!(template_function);
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..OC_BLOCK {
                ret[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(&out[b * osb + l * osh + (k + kk) * osw + j + v as i64 * T::Vec::SIZE as i64] as *const _ as *const T)
                }; // prettier-ignore
            }
        }
        ret
    };
    let is0 = b * isb + l * step_height * ish + k * step_width * isw;
    for n in 0..kh {
        let is1 = is0 + n * ish;
        for m in 0..kw {
            let is2 = is1 + m * isw;
            for i in ii..i_end {
                let is3 = is2 + i;
                let inp = conv2d_microkernel_gen_inps!(
                    inp,
                    is3,
                    step_width * isw,
                    template_function
                );
                let kernel_vecs = conv2d_microkernel_gen_kernels!(kernel, template_function);
                conv2d_microkernel_gen_results!(results, inp, kernel_vecs, template_function);
                kernel.add(OC_BLOCK * T::Vec::SIZE);
            }
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..OC_BLOCK {
            let out_vec = &mut out
                [b * osb + l * osh + (k + kk) * osw + j + (v * T::Vec::SIZE) as i64]
                as *mut _ as *mut T::Vec; // prettier-ignore
            unsafe {
                out_vec.write_unaligned(results[v as usize][kk as usize]);
            }
        }
    }
}

/// This struct carries the micro kernel function and the corresponding info
pub struct ConvKernel<T: CommonBounds> {
    pub(crate) kernel: fn(
        [i64; 2],
        [i64; 2],
        [i64; 4],
        [i64; 3],
        [i64; 2],
        [i64; 3],
        &mut Pointer<T>,
        &Pointer<T>,
        &mut Pointer<T>
    ),
    pub(crate) oc_block: usize,
    pub(crate) ow_block: usize,
}

impl<T: CommonBounds> ConvKernel<T> {
    pub(crate) fn new(
        kernel: fn(
            [i64; 2],
            [i64; 2],
            [i64; 4],
            [i64; 3],
            [i64; 2],
            [i64; 3],
            &mut Pointer<T>,
            &Pointer<T>,
            &mut Pointer<T>
        ),
        oc_block: usize,
        ow_block: usize
    ) -> Self {
        Self { kernel, oc_block, ow_block }
    }
    pub(crate) fn register_used(&self) -> usize {
        let res_used = self.oc_block * self.ow_block;
        let inp_used = self.ow_block;
        let kernel_used = 1;
        res_used + inp_used + kernel_used
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
    [ii, i_end]: [i64; 2],
    [kh, kw]: [i64; 2],
    [b, l, k, j]: [i64; 4],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    oc_end: i64,
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    kernel: &mut Pointer<T>
) {
    conv2d_microkernel_declare_const!(template_function);
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; 1];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..oc_end {
                ret[0][kk as usize][v as usize] = out[b * osb + l * osh + (k + kk) * osw + j + v];
            }
        }
        ret
    };
    let is0 = b * isb + l * step_height * ish + k * step_width * isw;
    for n in 0..kh {
        let is1 = is0 + n * ish;
        for m in 0..kw {
            let is2 = is1 + m * isw;
            for i in ii..i_end {
                let is3 = is2 + i;
                let inp = conv2d_microkernel_gen_inps!(
                    inp,
                    is3,
                    step_width * isw,
                    template_function
                );
                let mut kernel0 = (T::Vec::splat(T::ZERO),);
                for v in 0..oc_end {
                    kernel0.0[v as usize] = kernel[v as usize];
                }
                conv2d_microkernel_gen_results!(results, inp, kernel0, template_function);
                kernel.add(oc_end as usize);
            }
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..oc_end {
            out[b * osb + l * osh + (k + kk) * osw + j + v] = results[0][kk as usize][v as usize];
        }
    }
}

pub(crate) fn iconv2d_full_oc_kernel_dispatch<T: CommonBounds>(
    oc: &mut usize, // output channels block size
    kb: &mut usize // outwidth block size
) -> Option<ConvKernel<T>> {
    let kernels: [
        [
            fn(
                [i64; 2],
                [i64; 2],
                [i64; 4],
                [i64; 3],
                [i64; 2],
                [i64; 3],
                &mut Pointer<T>,
                &Pointer<T>,
                &mut Pointer<T>
            );
            5
        ];
        4
    ] = [
        [micro_kernel_1x1, micro_kernel_2x1, micro_kernel_3x1, micro_kernel_4x1, micro_kernel_5x1],
        [micro_kernel_1x2, micro_kernel_2x2, micro_kernel_3x2, micro_kernel_4x2, micro_kernel_5x2],
        [micro_kernel_1x4, micro_kernel_2x4, micro_kernel_3x4, micro_kernel_4x4, micro_kernel_5x4],
        [micro_kernel_1x8, micro_kernel_2x8, micro_kernel_3x8, micro_kernel_4x8, micro_kernel_5x8],
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

    let kernel_fn = kernels
        .get(map_oc)
        .map(|x| x.get(map_kb))
        .flatten();

    // println!("picked iconv2d_microkernel_{}x{} at {}{}", kb, oc, map_oc, map_kb);

    kernel_fn.cloned().map(|kernel| ConvKernel::new(kernel, *oc, *kb))
}

pub(crate) fn full_oc_kernels<T: CommonBounds>() -> [ConvKernel<T>; 20] {
    [
        ConvKernel::new(micro_kernel_1x1, 1, 1),
        ConvKernel::new(micro_kernel_2x1, 1, 2),
        ConvKernel::new(micro_kernel_3x1, 1, 3),
        ConvKernel::new(micro_kernel_4x1, 1, 4),
        ConvKernel::new(micro_kernel_5x1, 1, 5),
        ConvKernel::new(micro_kernel_1x2, 2, 1),
        ConvKernel::new(micro_kernel_2x2, 2, 2),
        ConvKernel::new(micro_kernel_3x2, 2, 3),
        ConvKernel::new(micro_kernel_4x2, 2, 4),
        ConvKernel::new(micro_kernel_5x2, 2, 5),
        ConvKernel::new(micro_kernel_1x4, 4, 1),
        ConvKernel::new(micro_kernel_2x4, 4, 2),
        ConvKernel::new(micro_kernel_3x4, 4, 3),
        ConvKernel::new(micro_kernel_4x4, 4, 4),
        ConvKernel::new(micro_kernel_5x4, 4, 5),
        ConvKernel::new(micro_kernel_1x8, 8, 1),
        ConvKernel::new(micro_kernel_2x8, 8, 2),
        ConvKernel::new(micro_kernel_3x8, 8, 3),
        ConvKernel::new(micro_kernel_4x8, 8, 4),
        ConvKernel::new(micro_kernel_5x8, 8, 5),
    ]
}

pub(crate) fn iconv2d_remain_oc_kernel_dispatch<T: CommonBounds>(
    kb: &mut usize // outwidth block size
) -> Option<
    fn(
        [i64; 2],
        [i64; 2],
        [i64; 4],
        [i64; 3],
        [i64; 2],
        [i64; 3],
        i64,
        &mut Pointer<T>,
        &Pointer<T>,
        &mut Pointer<T>
    )
> {
    let kernels: [
        fn(
            [i64; 2],
            [i64; 2],
            [i64; 4],
            [i64; 3],
            [i64; 2],
            [i64; 3],
            i64,
            &mut Pointer<T>,
            &Pointer<T>,
            &mut Pointer<T>
        );
        5
    ] = [micro_kernel_1_1, micro_kernel_2_1, micro_kernel_3_1, micro_kernel_4_1, micro_kernel_5_1];

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
