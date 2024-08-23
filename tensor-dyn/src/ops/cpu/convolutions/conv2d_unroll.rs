use crate::ops::cpu::vector::traits::Init;
use crate::ops::cpu::vector::traits::VecTrait;
use crate::tensor_base::_Tensor;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;

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
                std::ptr::copy(
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
pub fn conv2d_ex<T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign, const REGNUM: usize, const VECSIZE: usize, VEC>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<T>> where VEC: VecTrait<T> + Copy + Init<T>, T: IntoScalar<T> {

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
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
        img.pad(&[(ph_start, ph_end), (pw_start, pw_end), (0, 0)], T::ZERO)?
    } else {
        img.clone()
    };
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
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

    let oc_r8 = out_channels % VECSIZE as i64;
    if oc_r8 > 0 {
        let o_n = out_channels / VECSIZE as i64;
        let ow_r14 = out_width % REGNUM as i64;
        if ow_r14 > 0 {
            let ow_n = out_width / REGNUM as i64;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [VEC::splat(T::ZERO); REGNUM];
                    let mut res_vectors_heap = vec![VEC::splat(T::ZERO); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut T; REGNUM];
                    let mut res_ptrs_heap = vec![0 as *mut T; ow_r14 as usize];
                    let mut kernel_vector = VEC::splat(T::ZERO);
                    for l in 0..out_height {
                        for kp in 0..ow_n {
                            prepare_regs::<T, VEC, REGNUM, _>(
                                VECSIZE, REGNUM as i64, kp, &mut res_vectors, &mut res_ptrs, 
                                |_k|&mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<T, VEC, _>(
                                            VECSIZE,
                                            REGNUM as i64,
                                            &kernel[i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0] as *const T,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..REGNUM as i64 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k as usize].as_ptr(),
                                        res_ptrs[k as usize],
                                        VECSIZE,
                                    );
                                }
                            }
                        }
                        for kp in ow_n..ow_n + 1 {
                            prepare_regs::<T, VEC, REGNUM, _>(
                                VECSIZE, ow_r14, kp, &mut res_vectors_heap, res_ptrs_heap.as_mut_slice(), 
                                |_k|&mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<T, VEC, _>(
                                            VECSIZE,
                                            ow_r14,
                                            &kernel[i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0] as *const T,
                                            &mut kernel_vector,
                                            &mut res_vectors_heap,
                                            |k| inp[i * is2 + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..ow_r14 {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors_heap[k as usize].as_ptr(),
                                        res_ptrs_heap[k as usize],
                                        VECSIZE,
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
                        let mut res_vectors = vec![vec![T::ZERO; oc_r8 as usize]; REGNUM];
                        let mut res_vectors_heap =
                            vec![vec![T::ZERO; oc_r8 as usize]; ow_r14 as usize];
                        let mut res_ptrs = [0 as *mut T; REGNUM];
                        let mut res_ptrs_heap = vec![0 as *mut T; ow_r14 as usize];
                        let mut kernel_vector = vec![T::ZERO; oc_r8 as usize];
                        for kp in 0..ow_n {
                            prepare_regs2::<T, VEC, REGNUM, _>(
                                oc_r8 as usize,
                                REGNUM as i64,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0]
                                            as *const T; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..REGNUM as i64 {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1
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
                            for k in 0..REGNUM as i64 {
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
                            prepare_regs2::<T, VEC, REGNUM, _>(
                                oc_r8 as usize,
                                ow_r14,
                                kp,
                                &mut res_vectors_heap,
                                res_ptrs_heap.as_mut_slice(),
                                |_k| &mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0]
                                            as *const T; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..ow_r14 {
                                            let res_vector = &mut res_vectors_heap[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1
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
            let kp_end = out_width / REGNUM as i64;
            let o_n = out_channels / VECSIZE as i64;
            (0..o_n).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [VEC::splat(T::ZERO); REGNUM];
                    let mut res_ptrs = [0 as *mut T; REGNUM];
                    let mut kernel_vector = VEC::splat(T::ZERO);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<T, VEC, REGNUM, _>(
                                VECSIZE, REGNUM as i64, kp, &mut res_vectors, res_ptrs.as_mut_slice(), 
                                |_k|&mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<T, VEC, _>(
                                            VECSIZE,
                                            REGNUM as i64,
                                            &kernel[i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0] as *const T,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..REGNUM {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        VECSIZE,
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
                        let mut res_vectors = vec![vec![T::ZERO; oc_r8 as usize]; REGNUM];
                        let mut res_ptrs = [0 as *mut T; REGNUM];
                        let mut kernel_vector = vec![T::ZERO; oc_r8 as usize];
                        for kp in 0..kp_end {
                            prepare_regs2::<T, VEC, REGNUM, _>(
                                oc_r8 as usize,
                                REGNUM as i64,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel
                                            [i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0]
                                            as *const T; // prettier-ignore
                                        kernel_vector.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(kernel_ptr, oc_r8 as usize)
                                        });
                                        for k in 0..REGNUM as i64 {
                                            let res_vector = &mut res_vectors[k as usize];
                                            let i_val = inp[i * is2
                                                + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1
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
                            for k in 0..REGNUM {
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
        let ow_r14 = out_width % REGNUM as i64;
        if ow_r14 > 0 {
            let jp_end = out_channels / VECSIZE as i64;
            let kp_end = out_width / REGNUM as i64;
            let factor = 1;
            let oh_end = out_height / factor;
            (0..oh_end).into_par_iter().for_each_init(
                || out,
                |out, oh_end| {
                    let mut res_vectors = [VEC::splat(T::ZERO); REGNUM];
                    let mut res_vectors_heap = vec![VEC::splat(T::ZERO); ow_r14 as usize];
                    let mut res_ptrs = [0 as *mut T; REGNUM];
                    let mut res_ptrs_heap = vec![0 as *mut T; ow_r14 as usize];
                    let mut kernel_vector = VEC::splat(T::ZERO);
                    let out = out.ptr;
                    for jp in 0..jp_end {
                        for l in 0..factor {
                            let l = oh_end * factor + l;
                            for kp in 0..kp_end {
                                prepare_regs!(
                                    REGNUM as i64,
                                    VECSIZE as i64,
                                    [jp * VECSIZE as i64, kp * REGNUM as i64, l],
                                    [os0, os1, os2],
                                    [out, res_vectors, res_ptrs]
                                );
                                let ii = 1;
                                let i = VECSIZE as i64;
                                __kernel!(
                                    [T, VEC, VECSIZE, REGNUM as i64],
                                    [kernel_height, kernel_width, ii, i],
                                    [ks0, ks1, ks2, ks3],
                                    [is0, is1, is2],
                                    [jp * VECSIZE as i64, kp * REGNUM as i64, l, step_width, step_height, dw, dh],
                                    [kernel, kernel_vector, res_vectors, inp]
                                );
                                flush!(VECSIZE, REGNUM, res_vectors, res_ptrs);
                            }
                            for kp in kp_end..kp_end + 1 {
                                prepare_regs!(
                                    ow_r14,
                                    VECSIZE as i64,
                                    [jp * VECSIZE as i64, kp * REGNUM as i64, l],
                                    [os0, os1, os2],
                                    [out, res_vectors_heap, res_ptrs_heap]
                                );
                                let ii = 4;
                                let i = 2;
                                __kernel!(
                                    [T, VEC, VECSIZE, ow_r14],
                                    [kernel_height, kernel_width, ii, i],
                                    [ks0, ks1, ks2, ks3],
                                    [is0, is1, is2],
                                    [jp * VECSIZE as i64, kp * REGNUM as i64, l, step_width, step_height, dw, dh],
                                    [kernel, kernel_vector, res_vectors_heap, inp]
                                );
                                flush!(VECSIZE, ow_r14 as usize, res_vectors_heap, res_ptrs_heap);
                            }
                        }
                    }
                }
            );
        } else {
            let jp_end = out_channels / VECSIZE as i64;
            let kp_end = out_width / REGNUM as i64;
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [VEC::splat(T::ZERO); REGNUM];
                    let mut res_ptrs = [0 as *mut T; REGNUM];
                    let mut kernel_vector = VEC::splat(T::ZERO);
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            prepare_regs::<T, VEC, REGNUM, _>(
                                VECSIZE,
                                REGNUM as i64,
                                kp,
                                &mut res_vectors,
                                res_ptrs.as_mut_slice(),
                                |_k| &mut out[jp * VECSIZE as i64 * os2 + _k * os1 + l * os0]
                            );
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        micro_kernel::<T, VEC, _>(
                                            VECSIZE,
                                            REGNUM as i64,
                                            &kernel
                                                [
                                                    i * ks2 + jp * VECSIZE as i64 * ks3 + m * ks1 + n * ks0
                                                ] as *const T,
                                            &mut kernel_vector,
                                            &mut res_vectors,
                                            |k| inp[i * is2 + ((kp * REGNUM as i64 + k) * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0] // prettier-ignore
                                        );
                                    }
                                }
                            }
                            for k in 0..REGNUM {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        VECSIZE
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
pub(crate) fn calculate_valid_k_range(
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

pub(crate) fn calculate_valid_n_range(
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
pub(crate) fn micro_kernel<T: CommonBounds + IntoScalar<T>, VEC: VecTrait<T> + Copy + Init<T>, F>(
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
pub(crate) fn micro_kernel_range<T: CommonBounds + IntoScalar<T>, VEC: VecTrait<T> + Copy + Init<T>, F>(
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
pub(crate) fn prepare_regs<T: CommonBounds, VEC: VecTrait<T>, const REGN: usize, F>(
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
pub(crate) fn prepare_regs2<T: CommonBounds, U, const REGN: usize, F>(
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

#[allow(unused)]
pub(crate) fn find_divisors(n: i64) -> Vec<i64> {
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

#[allow(unused)]
pub(crate) fn get_cache_set(address: usize, cache_line_size: usize, num_cache_sets: usize) -> usize {
    // 计算块偏移 (Block Offset)，即低于缓存行大小的地址位
    let block_offset_bits = cache_line_size.trailing_zeros() as usize;

    // 获取缓存组索引 (Cache Set Index)
    (address >> block_offset_bits) % num_cache_sets
}