use crate::ops::cpu::vector::traits::Init;
use crate::ops::cpu::vector::traits::VecTrait;
use crate::slice::SliceOps;
use crate::tensor_base::_Tensor;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_common::slice::Slice;
use tensor_macros::match_selection;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;

use tensor_types::dtype::TypeCommon;

macro_rules! __kernel {
    (
        [$T:ident, $vec:ident, $vec_size:expr, $reg_num:expr],
        [$kh:ident, $kw:ident, $ii:ident, $i:ident],
        [$ks0:ident, $ks1:ident, $ks2:ident, $ks3:ident],
        [$is0:ident, $is1:ident, $is2:ident],
        [$jp:expr, $kp:expr, $l:ident, $step_width:ident, $step_height:ident, $dw:ident, $dh:ident],
        [$kernel:ident, $kvec:ident, $rvec:ident, $inp:ident]
    ) => {
        for __ii in 0..$ii {
            for n in 0..$kh {
                for m in 0..$kw {
                    for __i in 0..$i {
                        let i = __ii * $i + __i;
                        micro_kernel::<$T, $vec, _>(
                            $vec_size,
                            $reg_num,
                            &$kernel[i * $ks2 + $jp * $ks3 + m * $ks1 + n * $ks0] as *const $T,
                            &mut $kvec,
                            &mut $rvec,
                            |k| $inp[i * $is2 + (($kp + k) * $step_width + m * $dw) * $is1 + ($l * $step_height + n * $dh) * $is0]
                        );
                    }
                }
            }
        }
    };
}

macro_rules! flush {
    ($vec_size:expr, $reg_num:expr, $rvec:ident, $res_ptrs:expr) => {
        for k in 0..$reg_num {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    $rvec[k].as_ptr(),
                    $res_ptrs[k],
                    $vec_size,
                );
            }
        }
    };
}

macro_rules! prepare_regs {
    (
        $end:expr,
        $vec_size:expr,
        [$jp:expr, $kp:expr, $l:ident],
        [$os0:ident, $os1:ident, $os2:ident],
        [$out:ident, $res_vectors:ident, $res_ptrs:ident]
    ) => {
        for k in 0..$end {
            let ptr = unsafe { $out.offset(($jp * $os2 + ($kp + k) * $os1 + $l * $os0) as isize) };
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, $res_vectors[k as usize].as_mut_ptr(), 8);
            }
            $res_ptrs[k as usize] = ptr;
        }
    };
}

#[cfg(target_feature = "fma")]
pub fn conv2d_ex_f32(
    img: &_Tensor<f32>,
    kernels: &_Tensor<f32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<f32>> {
    use tensor_common::slice;
    use wide::f32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((pw_start, pw_end), (ph_start, ph_end)) = (padding[0], padding[1]);
    let (dw, dh) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
        let img_padded = _Tensor::<f32>::zeros([
            img_height + ph_start + ph_end,
            img_width + pw_start + pw_end,
            img_channels,
        ])?;
        let he = img_height + ph_start;
        let we = img_width + pw_start;
        let mut slice = slice!(img_padded[ph_start:he, pw_start:we, :])?;
        slice.assign(&img);
        img_padded
    } else {
        img.clone()
    };
    let output = _Tensor::<f32>::zeros([out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let oc_r8 = out_channels % 8;
    if oc_r8 > 0 {
        let o_n = out_channels / 8;
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let ow_n = out_width / 14;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(0.0); 14];
                    let mut res_vectors_heap = vec![f32x8::splat(0.0); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                    let mut kernel_vector = f32x8::splat(0.0);
                    for l in 0..out_height {
                        for kp in 0..ow_n {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, &mut res_ptrs, 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<f32, f32x8, _>(
                                            8,
                                            14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, ow_r14, kp, &mut res_vectors_heap, res_ptrs_heap.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<f32, f32x8, _>(
                                            8,
                                            ow_r14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![f32::ZERO; oc_r8 as usize]; 14];
                        let mut res_vectors_heap =
                            vec![vec![f32::ZERO; oc_r8 as usize]; ow_r14 as usize];
                        let mut res_ptrs = [0 as *mut f32; 14];
                        let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                        let mut kernel_vector = vec![f32::ZERO; oc_r8 as usize];
                        for kp in 0..ow_n {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..14i64 {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw) * is1
                                                + (l * step_height + n * dh) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..ow_r14 {
                                            let res_vector = &mut res_vectors_heap[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw) * is1
                                                + (l * step_height + n * dh) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        } else {
            let kp_end = out_width / 14;
            let o_n = out_channels / 8;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(0.0); 14];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut kernel_vector = f32x8::splat(0.0);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, res_ptrs.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<f32, f32x8, _>(
                                            8,
                                            14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![f32::ZERO; oc_r8 as usize]; 14];
                        let mut res_ptrs = [0 as *mut f32; 14];
                        let mut kernel_vector = vec![f32::ZERO; oc_r8 as usize];
                        for kp in 0..kp_end {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..14i64 {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw) * is1
                                                + (l * step_height + n * dh) * is0]; // prettier-ignore
                                            res_vector.iter_mut().enumerate().for_each(
                                                |(idx, val)| {
                                                    *val += unsafe {
                                                        *kernel_vector.as_ptr().wrapping_add(idx)
                                                    } * i_val;
                                                },
                                            ); // prettier-ignore
                                        }
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        }
    } else {
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            let factor = 2;
            let oh_end = out_height / factor;
            (0..oh_end).into_par_iter().for_each_init(
                || out,
                |out, oh_end| {
                    let mut res_vectors = [f32x8::splat(0.0); 14];
                    let mut res_vectors_heap = vec![f32x8::splat(0.); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                    let mut kernel_vector = f32x8::splat(0.0);
                    let out = out.ptr;
                    for jp in 0..jp_end {
                        for l in 0..factor {
                            let l = oh_end * factor + l;
                            for kp in 0..kp_end {
                                prepare_regs!(
                                    14,
                                    8,
                                    [jp * 8, kp * 14, l],
                                    [os0, os1, os2],
                                    [out, res_vectors, res_ptrs]
                                );
                                let ii = 4;
                                let i = 2;
                                __kernel!(
                                    [f32, f32x8, 8, 14],
                                    [kernel_height, kernel_width, ii, i],
                                    [ks0, ks1, ks2, ks3],
                                    [is0, is1, is2],
                                    [jp * 8, kp * 14, l, step_width, step_height, dw, dh],
                                    [kernel, kernel_vector, res_vectors, inp]
                                );
                                flush!(8, 14, res_vectors, res_ptrs);
                            }
                            for kp in kp_end..kp_end + 1 {
                                prepare_regs!(
                                    ow_r14,
                                    8,
                                    [jp * 8, kp * 14, l],
                                    [os0, os1, os2],
                                    [out, res_vectors_heap, res_ptrs_heap]
                                );
                                let ii = 4;
                                let i = 2;
                                __kernel!(
                                    [f32, f32x8, 8, ow_r14],
                                    [kernel_height, kernel_width, ii, i],
                                    [ks0, ks1, ks2, ks3],
                                    [is0, is1, is2],
                                    [jp * 8, kp * 14, l, step_width, step_height, dw, dh],
                                    [kernel, kernel_vector, res_vectors_heap, inp]
                                );
                                flush!(8, ow_r14 as usize, res_vectors_heap, res_ptrs_heap);
                            }
                        }
                    }
                }
            );
        } else {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(0.0); 14];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut kernel_vector = f32x8::splat(0.0);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<f32, f32x8, _>(
                                            8,
                                            14,
                                            &kernel
                                                [
                                                    i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0
                                                ] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        }
    }
    Ok(output)
}

#[cfg(target_feature = "fma")]
pub fn conv2d_ex_i32(
    img: &_Tensor<i32>,
    kernels: &_Tensor<i32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<i32>> {
    use tensor_common::slice;
    use wide::i32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((pw_start, pw_end), (ph_start, ph_end)) = (padding[0], padding[1]);
    let (dw, dh) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
        let img_padded = _Tensor::<i32>::zeros([
            img_height + ph_start + ph_end,
            img_width + pw_start + pw_end,
            img_channels,
        ])?;
        let he = img_height + ph_start;
        let we = img_width + pw_start;
        let mut slice = slice!(img_padded[ph_start:he, pw_start:we, :])?;
        slice.assign(&img);
        img_padded
    } else {
        img.clone()
    };
    let output = _Tensor::<i32>::zeros([out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let oc_r8 = out_channels % 8;
    if oc_r8 > 0 {
        let o_n = out_channels / 8;
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let ow_n = out_width / 14;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_vectors_heap = vec![i32x8::splat(0); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut i32; ow_r14 as usize];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        for kp in 0..ow_n {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, &mut res_ptrs, 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<i32, i32x8, _>(
                                            8,
                                            14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8, ow_r14, kp, &mut res_vectors_heap, res_ptrs_heap.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<i32, i32x8, _>(
                                            8,
                                            ow_r14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![i32::ZERO; oc_r8 as usize]; 14];
                        let mut res_vectors_heap =
                            vec![vec![i32::ZERO; oc_r8 as usize]; ow_r14 as usize];
                        let mut res_ptrs = [0 as *mut i32; 14];
                        let mut res_ptrs_heap = vec![0 as *mut i32; ow_r14 as usize];
                        let mut kernel_vector = vec![i32::ZERO; oc_r8 as usize];
                        for kp in 0..ow_n {
                            prepare_regs2::<i32, wide::i32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const i32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..14i64 {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw) * is1
                                                + (l * step_height + n * dh) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs2::<i32, wide::i32x8, 14, _>(
                                oc_r8 as usize,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const i32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..ow_r14 {
                                            let res_vector = &mut res_vectors_heap[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw) * is1
                                                + (l * step_height + n * dh) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        } else {
            let kp_end = out_width / 14;
            let o_n = out_channels / 8;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, res_ptrs.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<i32, i32x8, _>(
                                            8,
                                            14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![i32::ZERO; oc_r8 as usize]; 14];
                        let mut res_ptrs = [0 as *mut i32; 14];
                        let mut kernel_vector = vec![i32::ZERO; oc_r8 as usize];
                        for kp in 0..kp_end {
                            prepare_regs2::<i32, wide::i32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const i32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..14i64 {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw) * is1
                                                + (l * step_height + n * dh) * is0]; // prettier-ignore
                                            res_vector.iter_mut().enumerate().for_each(
                                                |(idx, val)| {
                                                    *val += unsafe {
                                                        *kernel_vector.as_ptr().wrapping_add(idx)
                                                    } * i_val;
                                                },
                                            ); // prettier-ignore
                                        }
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        }
    } else {
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_vectors_heap = vec![i32x8::splat(0); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut i32; ow_r14 as usize];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<i32, i32x8, _>(
                                            8,
                                            14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                        for kp in kp_end..kp_end + 1 {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<i32, i32x8, _>(
                                            8,
                                            ow_r14,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        } else {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<i32, i32x8, _>(
                                            8,
                                            14,
                                            &kernel
                                                [
                                                    i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0
                                                ] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        }
    }
    Ok(output)
}

#[cfg(target_feature = "fma")]
pub fn conv2d_ex_i32_enhanced(
    img: &_Tensor<i32>,
    kernels: &_Tensor<i32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<i32>> {
    use wide::i32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((pw_start, pw_end), (ph_start, ph_end)) = (padding[0], padding[1]);
    let (dw, dh) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let output = _Tensor::<i32>::zeros([out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let oc_r8 = out_channels % 8;
    if oc_r8 > 0 {
        let o_n = out_channels / 8;
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let ow_n = out_width / 14;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_vectors_heap = vec![i32x8::splat(0); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut i32; ow_r14 as usize];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..ow_n {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, &mut res_ptrs, 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<i32, i32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8, ow_r14, kp, &mut res_vectors_heap, res_ptrs_heap.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        ow_r14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<i32, i32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![i32::ZERO; oc_r8 as usize]; 14];
                        let mut res_vectors_heap =
                            vec![vec![i32::ZERO; oc_r8 as usize]; ow_r14 as usize];
                        let mut res_ptrs = [0 as *mut i32; 14];
                        let mut res_ptrs_heap = vec![0 as *mut i32; ow_r14 as usize];
                        let mut kernel_vector = vec![i32::ZERO; oc_r8 as usize];
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..ow_n {
                            prepare_regs2::<i32, wide::i32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const i32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1
                                                + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs2::<i32, wide::i32x8, 14, _>(
                                oc_r8 as usize,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        ow_r14
                                    );
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                                [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                                as *const i32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let _k = kp * 14 + k;
                                            let res_vector = &mut res_vectors_heap[k as usize];
                                            let i_val = inp[i * is2
                                                        + (_k * step_width + m * dw - pw_start) * is1
                                                        + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        } else {
            let kp_end = out_width / 14;
            let o_n = out_channels / 8;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..kp_end {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, res_ptrs.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<i32, i32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![i32::ZERO; oc_r8 as usize]; 14];
                        let mut res_ptrs = [0 as *mut i32; 14];
                        let mut kernel_vector = vec![i32::ZERO; oc_r8 as usize];
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..kp_end {
                            prepare_regs2::<i32, wide::i32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const i32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1
                                                + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector.iter_mut().enumerate().for_each(
                                                |(idx, val)| {
                                                    *val += unsafe {
                                                        *kernel_vector.as_ptr().wrapping_add(idx)
                                                    } * i_val;
                                                },
                                            ); // prettier-ignore
                                        }
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        }
    } else {
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_vectors_heap = vec![i32x8::splat(0); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut i32; ow_r14 as usize];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..kp_end {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<i32, i32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                        for kp in kp_end..kp_end + 1 {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        ow_r14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<i32, i32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const i32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        } else {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [i32x8::splat(0); 14];
                    let mut res_ptrs = [0 as *mut i32; 14];
                    let mut kernel_vector = i32x8::splat(0);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..kp_end {
                            prepare_regs::<i32, wide::i32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );

                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<i32, i32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel
                                                [
                                                    i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0
                                                ] as *const i32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        }
    }
    Ok(output)
}

#[cfg(target_feature = "fma")]
pub fn conv2d_ex_f32_enhanced(
    img: &_Tensor<f32>,
    kernels: &_Tensor<f32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<f32>> {
    use wide::f32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((pw_start, pw_end), (ph_start, ph_end)) = (padding[0], padding[1]);
    let (dw, dh) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let output = _Tensor::<f32>::zeros([out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let oc_r8 = out_channels % 8;
    if oc_r8 > 0 {
        let o_n = out_channels / 8;
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let ow_n = out_width / 14;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_vectors_heap = vec![f32x8::splat(f32::ZERO); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..ow_n {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, &mut res_ptrs, 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, ow_r14, kp, &mut res_vectors_heap, res_ptrs_heap.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        ow_r14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![f32::ZERO; oc_r8 as usize]; 14];
                        let mut res_vectors_heap =
                            vec![vec![f32::ZERO; oc_r8 as usize]; ow_r14 as usize];
                        let mut res_ptrs = [0 as *mut f32; 14];
                        let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                        let mut kernel_vector = vec![f32::ZERO; oc_r8 as usize];
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..ow_n {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1
                                                + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, ow_r14); // prettier-ignore
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let _k = kp * 14 + k;
                                            let res_vector = &mut res_vectors_heap[k as usize];
                                            let i_val = inp[i * is2
                                                        + (_k * step_width + m * dw - pw_start) * is1
                                                        + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        } else {
            let kp_end = out_width / 14;
            let o_n = out_channels / 8;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, res_ptrs.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![f32::ZERO; oc_r8 as usize]; 14];
                        let mut res_ptrs = [0 as *mut f32; 14];
                        let mut kernel_vector = vec![f32::ZERO; oc_r8 as usize];
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1
                                                + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector.iter_mut().enumerate().for_each(
                                                |(idx, val)| {
                                                    *val += unsafe {
                                                        *kernel_vector.as_ptr().wrapping_add(idx)
                                                    } * i_val;
                                                },
                                            ); // prettier-ignore
                                        }
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        }
    } else {
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_vectors_heap = vec![f32x8::splat(f32::ZERO); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                        for kp in kp_end..kp_end + 1 {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, ow_r14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        } else {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );

                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        }
    }
    Ok(output)
}

#[cfg(target_feature = "fma")]
pub fn conv2d_ex_f32_enhanced_block(
    img: &_Tensor<f32>,
    kernels: &_Tensor<f32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<f32>> {
    use wide::f32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((pw_start, pw_end), (ph_start, ph_end)) = (padding[0], padding[1]);
    let (dw, dh) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let output = _Tensor::<f32>::zeros([out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let oc_r8 = out_channels % 8;
    if oc_r8 > 0 {
        let o_n = out_channels / 8;
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let ow_n = out_width / 14;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_vectors_heap = vec![f32x8::splat(f32::ZERO); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..ow_n {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, &mut res_ptrs, 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, ow_r14, kp, &mut res_vectors_heap, res_ptrs_heap.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        ow_r14
                                    );
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![f32::ZERO; oc_r8 as usize]; 14];
                        let mut res_vectors_heap =
                            vec![vec![f32::ZERO; oc_r8 as usize]; ow_r14 as usize];
                        let mut res_ptrs = [0 as *mut f32; 14];
                        let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                        let mut kernel_vector = vec![f32::ZERO; oc_r8 as usize];
                        let (kh_start, kh_end) = calculate_valid_n_range(
                            l,
                            step_height,
                            ph_start,
                            dh,
                            img_height,
                            kernel_height
                        );
                        for kp in 0..ow_n {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(
                                        kp,
                                        m,
                                        dw,
                                        step_width,
                                        pw_start,
                                        img_width,
                                        14
                                    );
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1
                                                + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, ow_r14); // prettier-ignore
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let _k = kp * 14 + k;
                                            let res_vector = &mut res_vectors_heap[k as usize];
                                            let i_val = inp[i * is2
                                                        + (_k * step_width + m * dw - pw_start) * is1
                                                        + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector
                                                .iter_mut()
                                                .zip(kernel_vector.iter())
                                                .for_each(|(res, ker)| {
                                                    *res += i_val * *ker;
                                                });
                                        }
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        } else {
            let kp_end = out_width / 14;
            let o_n = out_channels / 8;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8, 14, kp, &mut res_vectors, res_ptrs.as_mut_slice(), 
                                |_k|&mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8,
                                    );
                                }
                            }
                        }
                    }
                },
            ); // prettier-ignore
            for jp in o_n..o_n + 1 {
                (0..out_height).into_par_iter().for_each_init(
                    || out,
                    |out, l| {
                        let mut res_vectors = vec![vec![f32::ZERO; oc_r8 as usize]; 14];
                        let mut res_ptrs = [0 as *mut f32; 14];
                        let mut kernel_vector = vec![f32::ZERO; oc_r8 as usize];
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs2::<f32, wide::f32x8, 14, _>(
                                oc_r8 as usize,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0]
                                            as *const f32; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in start..end {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1
                                                + (l * step_height + n * dh - ph_start) * is0]; // prettier-ignore
                                            res_vector.iter_mut().enumerate().for_each(
                                                |(idx, val)| {
                                                    *val += unsafe {
                                                        *kernel_vector.as_ptr().wrapping_add(idx)
                                                    } * i_val;
                                                },
                                            ); // prettier-ignore
                                        }
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        oc_r8 as usize
                                    );
                                }
                            }
                        }
                    }
                );
            }
        }
    } else {
        let ow_r14 = out_width % 14;
        if ow_r14 > 0 {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_vectors_heap = vec![f32x8::splat(f32::ZERO); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut res_ptrs_heap = vec![0 as *mut f32; ow_r14 as usize];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                        for kp in kp_end..kp_end + 1 {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, ow_r14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        } else {
            let jp_end = out_channels / 8;
            let kp_end = out_width / 14;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); 14];
                    let mut res_ptrs = [0 as *mut f32; 14];
                    let mut kernel_vector = f32x8::splat(f32::ZERO);
                    for l in 0..out_height {
                        let (kh_start, kh_end) = calculate_valid_n_range(l, step_height, ph_start, dh, img_height, kernel_height); // prettier-ignore
                        for kp in 0..kp_end {
                            prepare_regs::<f32, wide::f32x8, 14, _>(
                                8,
                                14,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * 8i64 * os2 + _k * os1 + l * os0]
                            );

                            for n in kh_start..kh_end {
                                for m in 0..kernel_width {
                                    let (start, end) = calculate_valid_k_range(kp, m, dw, step_width, pw_start, img_width, 14); // prettier-ignore
                                    for i in 0..in_channels {
                                        micro_kernel_range::<f32, f32x8, _>(
                                            start,
                                            end,
                                            8,
                                            &kernel[i * ks2 + jp * 8i64 * ks3 + m * ks1 + n * ks0] as *const f32, // prettier-ignore
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * 14i64 + k) * step_width + m * dw - pw_start) * is1 + (l * step_height + n * dh - ph_start) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            );
        }
    }
    Ok(output)
}

#[inline(always)]
fn calculate_valid_k_range(
    kp: i64,
    m: i64,
    dw: i64,
    step_width: i64,
    pw_start: i64,
    img_width: i64,
    end: i64
) -> (i64, i64) {
    let base = m * dw + kp * 14 * step_width - pw_start;

    let k_start = (-base + step_width - 1) / step_width;
    let k_end = (img_width - base + step_width - 1) / step_width;

    (k_start.max(0), k_end.min(end))
}

fn calculate_valid_n_range(
    l: i64,
    step_height: i64,
    ph_start: i64,
    dh: i64,
    img_height: i64,
    kernel_height: i64
) -> (i64, i64) {
    let base = l * step_height - ph_start;

    let n_start = (-base + dh - 1) / dh;
    let n_end = (img_height - base) / dh;

    (n_start.max(0), n_end.min(kernel_height))
}

#[inline(always)]
fn micro_kernel<T: CommonBounds + IntoScalar<T>, VEC: VecTrait<T> + Copy + Init<T>, F>(
    vec_size: usize,
    reg_num: i64,
    kernel_ptr: *const T,
    kvec: &mut VEC,
    rvec: &mut [VEC],
    f: F
)
    where F: Fn(i64) -> T
{
    kvec.copy_from_slice(unsafe { std::slice::from_raw_parts(kernel_ptr, vec_size) });
    let remain = reg_num % 4;
    let k_end = reg_num - remain;
    for k in 0..k_end / 4 {
        let res_vector = &mut rvec[(k * 4) as usize];
        *res_vector = kvec._mul_add(VEC::splat(f(k * 4).into_scalar()), *res_vector);
        let res_vector = &mut rvec[((k * 4) as usize) + 1];
        *res_vector = kvec._mul_add(VEC::splat(f(k * 4 + 1).into_scalar()), *res_vector);
        let res_vector = &mut rvec[((k * 4) as usize) + 2];
        *res_vector = kvec._mul_add(VEC::splat(f(k * 4 + 2).into_scalar()), *res_vector);
        let res_vector = &mut rvec[((k * 4) as usize) + 3];
        *res_vector = kvec._mul_add(VEC::splat(f(k * 4 + 3).into_scalar()), *res_vector);
    }
    for k in k_end..reg_num {
        let res_vector = &mut rvec[k as usize];
        *res_vector = kvec._mul_add(VEC::splat(f(k).into_scalar()), *res_vector);
    }
}

#[inline(always)]
fn micro_kernel_range<T: CommonBounds + IntoScalar<T>, VEC: VecTrait<T> + Copy + Init<T>, F>(
    start: i64,
    end: i64,
    vec_size: usize,
    kernel_ptr: *const T,
    kvec: &mut VEC,
    rvec: &mut [VEC],
    f: F
)
    where F: Fn(i64) -> T
{
    kvec.copy_from_slice(unsafe { std::slice::from_raw_parts(kernel_ptr, vec_size) });
    for k in start..end {
        let res_vector = &mut rvec[k as usize];
        res_vector.fma(*kvec, VEC::splat(f(k).into_scalar()));
    }
}

#[inline(always)]
fn prepare_regs<T: CommonBounds, VEC: VecTrait<T>, const REGN: usize, F>(
    vec_size: usize,
    reg_num: i64,
    kp: i64,
    res_vectors: &mut [VEC],
    res_ptrs: &mut [*mut T],
    mut f: F
)
    where F: FnMut(i64) -> *mut T
{
    for k in 0..reg_num {
        let _k = kp * (REGN as i64) + k;
        let ptr = f(_k);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, res_vectors[k as usize].as_mut_ptr(), vec_size);
        }
        res_ptrs[k as usize] = ptr;
    }
}

#[inline(always)]
fn prepare_regs2<T: CommonBounds, U, const REGN: usize, F>(
    vec_size: usize,
    reg_num: i64,
    kp: i64,
    res_vectors: &mut [Vec<T>],
    res_ptrs: &mut [*mut T],
    mut f: F
)
    where F: FnMut(i64) -> *mut T
{
    for k in 0..reg_num {
        let _k = kp * (REGN as i64) + k;
        let res_vec = unsafe { std::slice::from_raw_parts_mut(f(_k), vec_size) }; // prettier-ignore
        res_vectors[k as usize].copy_from_slice(res_vec);
        res_ptrs[k as usize] = res_vec.as_mut_ptr();
    }
}

fn find_divisors(n: i64) -> Vec<i64> {
    let mut divisors = Vec::new();

    for i in 1..=(n as f64).sqrt() as i64 {
        if n % i == 0 {
            divisors.push(i);
            if i != n / i {
                divisors.push(n / i);
            }
        }
    }

    divisors.sort();
    divisors
}
