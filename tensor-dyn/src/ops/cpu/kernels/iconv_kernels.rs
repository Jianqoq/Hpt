use duplicate::duplicate_item;
use tensor_common::pointer::Pointer;
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
    ($name:ident, $kr3:expr, $vec_size:expr, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                T::Vec::splat($name[$kr3 + $idx * $vec_size]),
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

macro_rules! repeat_results_scalar {
    ($results:ident, $inp:ident, $kernel:ident, [$($idx:expr),*]) => {
        paste::paste! {
            $(
                $results[$idx] = $inp.$idx.mul_add($kernel, $results[$idx]);
            )*
        }
    };
}

duplicate::duplicate!([
    template_function   ow_block  oc_block      inp_place_holder            oc;
    [micro_kernel_5x1]    [5]      [1]          [[0, 1, 2, 3, 4]]           [[0]];
    [micro_kernel_4x1]    [4]      [1]          [[0, 1, 2, 3]]              [[0]];
    [micro_kernel_3x1]    [3]      [1]          [[0, 1, 2]]                 [[0]];
    [micro_kernel_2x1]    [2]      [1]          [[0, 1]]                    [[0]];
    [micro_kernel_1x1]    [1]      [1]          [[0]]                       [[0]];
    [micro_kernel_5x2]    [5]      [2]          [[0, 1, 2, 3, 4]]           [[0, 1]];
    [micro_kernel_4x2]    [4]      [2]          [[0, 1, 2, 3]]              [[0, 1]];
    [micro_kernel_3x2]    [3]      [2]          [[0, 1, 2]]                 [[0, 1]];
    [micro_kernel_2x2]    [2]      [2]          [[0, 1]]                    [[0, 1]];
    [micro_kernel_1x2]    [1]      [2]          [[0]]                       [[0, 1]];
    [micro_kernel_5x4]    [5]      [4]          [[0, 1, 2, 3, 4]]           [[0, 1, 2, 3]];
    [micro_kernel_4x4]    [4]      [4]          [[0, 1, 2, 3]]              [[0, 1, 2, 3]];
    [micro_kernel_3x4]    [3]      [4]          [[0, 1, 2]]                 [[0, 1, 2, 3]];
    [micro_kernel_2x4]    [2]      [4]          [[0, 1]]                    [[0, 1, 2, 3]];
    [micro_kernel_1x4]    [1]      [4]          [[0]]                       [[0, 1, 2, 3]];
    [micro_kernel_5x8]    [5]      [8]          [[0, 1, 2, 3, 4]]           [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_4x8]    [4]      [8]          [[0, 1, 2, 3]]              [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_3x8]    [3]      [8]          [[0, 1, 2]]                 [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_2x8]    [2]      [8]          [[0, 1]]                    [[0, 1, 2, 3, 4, 5, 6, 7]];
    [micro_kernel_1x8]    [1]      [8]          [[0]]                       [[0, 1, 2, 3, 4, 5, 6, 7]];]
#[inline]
fn template_function<T: CommonBounds>(
    [ii, i_end]: [i64; 2],
    [kh, kw]: [i64; 2],
    [b, l, k, j]: [i64; 4],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [ks0, ks1, ks2]: [i64; 3],
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    kernel: &Pointer<T>,
) {
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); ow_block]; oc_block]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); ow_block]; oc_block];
        for v in 0..oc_block {
            for kk in 0..ow_block as i64 {
                ret[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(
                        &out[b * osb
                            + l * osh
                            + (k + kk) * osw
                            + j
                            + v as i64 * T::Vec::SIZE as i64] as *const _
                            as *const T,
                    )
                }; // prettier-ignore
            }
        }
        ret
    };
    let is0 = b * isb + l * step_height * ish + k * step_width * isw;
    let kr0 = j;
    for n in 0..kh {
        let is1 = is0 + n * ish;
        let kr1 = n * ks0 + kr0;
        for m in 0..kw {
            let is2 = is1 + m * isw;
            let kr2 = kr1 + m * ks1;
            for i in ii..i_end {
                let is3 = is2 + i;
                let kr3 = i * ks2 + kr2;
                let inp = repeat_inp!(inp, is3, step_width * isw, inp_place_holder);
                let kernel = repeat_kernel!(kernel, kr3, T::Vec::SIZE as i64, oc);
                repeat_results!(results, inp, kernel, oc, inp_place_holder);
            }
        }
    }
    for v in 0..oc_block {
        for kk in 0..ow_block as i64 {
            let out_vec = &mut out[b * osb + l * osh + (k + kk) * osw + j + v * T::Vec::SIZE as i64]
                as *mut _ as *mut T::Vec; // prettier-ignore
            unsafe {
                out_vec.write_unaligned(results[v as usize][kk as usize]);
            }
        }
    }
}
);

#[duplicate_item(
        template_function   inp_place_holder      kernel_place_holder   oc;
        [micro_kernel_5]    [[0, 1, 2, 3, 4]]       [[0, 1]]           [[0]];
        [micro_kernel_4]    [[0, 1, 2, 3]]          [[0, 1]]           [[0]];
        [micro_kernel_3]    [[0, 1, 2]]             [[0, 1]]           [[0]];
        [micro_kernel_2]    [[0, 1]]                [[0, 1]]           [[0]];
        [micro_kernel_1]    [[0]]                   [[0, 1]]           [[0]];
)]
#[inline]
fn template_function<T: CommonBounds>(
    [ii, i_end]: [i64; 2],
    [kh, kw]: [i64; 2],
    [b, l, k, j]: [i64; 4],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [ks0, ks1, ks2]: [i64; 3],
    oc_end: i64,
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    kernel: &Pointer<T>
) {
    const OW_BLOCK: usize = 5;
    let mut results = if ii == 0 {
        [T::Vec::splat(T::ZERO); OW_BLOCK]
    } else {
        let mut ret = [T::Vec::splat(T::ZERO); OW_BLOCK];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..oc_end {
                ret[kk as usize][v as usize] = out[b * osb + l * osh + (k + kk) * osw + j];
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
                let inp = repeat_inp!(inp, is3, step_width * isw, inp_place_holder);
                let mut kernel0 = T::Vec::splat(T::ZERO);
                for v in 0..oc_end {
                    kernel0[v as usize] = kernel[n * ks0 + m * ks1 + i * ks2 + j + v];
                }
                repeat_results_scalar!(results, inp, kernel0, inp_place_holder);
            }
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..oc_end {
            out[b * osb + l * osh + (k + kk) * osw + j] = results[kk as usize][v as usize];
        }
    }
}

pub(crate) fn iconv2d_full_oc_kernel_dispatch<T: CommonBounds>(
    oc: &mut usize, // output channels block size
    kb: &mut usize // outwidth block size
) -> fn(
    [i64; 2],
    [i64; 2],
    [i64; 4],
    [i64; 3],
    [i64; 2],
    [i64; 3],
    [i64; 3],
    &mut Pointer<T>,
    &Pointer<T>,
    &Pointer<T>
) {
    let kernels: [
        [
            fn(
                [i64; 2],
                [i64; 2],
                [i64; 4],
                [i64; 3],
                [i64; 2],
                [i64; 3],
                [i64; 3],
                &mut Pointer<T>,
                &Pointer<T>,
                &Pointer<T>
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

    // println!("picked iconv2d_microkernel_{}x{} at {}{}", kb, oc, map_oc(oc), map_kb(kb));

    if let Some(kernel_fn) = kernel_fn {
        kernel_fn.clone()
    } else {
        panic!("unable to find iconv2d_microkernel_{}x{}", kb, oc);
    }
}

pub(crate) fn iconv2d_remain_oc_kernel_dispatch<T: CommonBounds>(
    kb: &mut usize // outwidth block size
) -> fn(
    [i64; 2],
    [i64; 2],
    [i64; 4],
    [i64; 3],
    [i64; 2],
    [i64; 3],
    [i64; 3],
    i64,
    &mut Pointer<T>,
    &Pointer<T>,
    &Pointer<T>
) {
    let kernels: [
        fn(
            [i64; 2],
            [i64; 2],
            [i64; 4],
            [i64; 3],
            [i64; 2],
            [i64; 3],
            [i64; 3],
            i64,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>
        );
        5
    ] = [micro_kernel_1, micro_kernel_2, micro_kernel_3, micro_kernel_4, micro_kernel_5];

    // println!("picked iconv2d_remain_microkernel_{} at {}", kb, map_kb(kb));
    let map_kb = map_kb(*kb);
    *kb = map_kb + 1;

    let kernel_fn = kernels.get(map_kb);

    if let Some(kernel_fn) = kernel_fn {
        kernel_fn.clone()
    } else {
        panic!("unable to find iconv2d_microkernel_remain_{}", kb);
    }
}

fn map_kb(kb: usize) -> usize {
    if kb <= 1 { 0 } else if kb <= 2 { 1 } else if kb <= 3 { 2 } else if kb <= 4 { 3 } else { 4 }
}

fn map_oc(oc: usize) -> usize {
    if oc <= 1 { 0 } else if oc <= 2 { 1 } else if oc <= 4 { 2 } else { 3 }
}
