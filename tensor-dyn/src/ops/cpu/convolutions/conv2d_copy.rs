
use crate::{ops::cpu::{convolutions::conv2d_unroll::{calculate_valid_k_range, calculate_valid_n_range, micro_kernel_range, prepare_regs, prepare_regs2}, vector::traits::VecTrait}, tensor_base::_Tensor};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tensor_traits::{TensorCreator, TensorInfo};
use wide::f32x8;
use tensor_types::dtype::TypeCommon;

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

pub fn conv2d_ex_f32_enhanced(
    img: &_Tensor<f32>,
    kernels: &_Tensor<f32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<f32>> {

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

pub fn conv2d_ex_f32_enhanced_block(
    img: &_Tensor<f32>,
    kernels: &_Tensor<f32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<f32>> {

    use crate::ops::cpu::convolutions::conv2d_unroll::{calculate_valid_k_range, calculate_valid_n_range, micro_kernel_range, prepare_regs};

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