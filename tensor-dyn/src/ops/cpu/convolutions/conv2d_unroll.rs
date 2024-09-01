use tensor_types::vectors::traits::*;
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
pub fn conv2d_ex_naive<
    T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign,
    const REGNUM: usize,
    const VECSIZE: usize,
    VEC
    >(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where VEC: VecTrait<T> + Copy + Init<T>, T: IntoScalar<T>
{
    let img_shape = img.shape();
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
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
        img.pad(
            &[
                (0, 0),
                (ph_start, ph_end),
                (pw_start, pw_end),
                (0, 0),
            ],
            T::ZERO
        )?
    } else {
        img.clone()
    };
    let output = _Tensor::<T>::zeros([batch, out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let mut kernel = kernels.ptr();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = output.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels

    let l2_cache = cache_size::l2_cache_size().unwrap_or(256 * 1024) / std::mem::size_of::<T>();

    let (co_b, wo_b, ci_b) = find_exact_combination(
        l2_cache as i64,
        out_channels as i64,
        out_width as i64,
        in_channels as i64
    );
    // co_b * vec_size * ci_b * 2 + ci_b * wo_b + wo_b * co_b * vec_size * 2 <= l2_cache
    let co_b_remain = co_b % (VECSIZE as i64);
    let co_b = co_b - co_b_remain;

    let num_co_b = out_channels / co_b;
    let num_wo_b = out_width / wo_b;
    let num_ci_b = in_channels / ci_b;

    for b in 0..batch {
        for _ in 0..num_co_b {
            for ip in 0..num_ci_b {
                for l in 0..out_height {
                    for kp in 0..num_wo_b {
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                for i in 0..ci_b {
                                    let i = ip * ci_b + i;
                                    for k in 0..wo_b {
                                        let k = kp * wo_b + k;
                                        for j in 0..co_b {
                                            out[b * osb + l * osh + k * osw + j] +=
                                            inp[b * isb + (l * step_height + n * dh) * ish + (k * step_width + m * dw) * isw + ip * ci_b + i] *
                                            kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j]; // prettier-ignore
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            out.offset(co_b);
            kernel.offset(co_b);
        }
    }

    Ok(output)
}
use num::traits::MulAdd;

#[cfg(target_feature = "fma")]
pub fn conv2d_ex<
    T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign + MulAdd<Output = T>,
    const REGNUM: usize,
    const VECSIZE: usize,
    VEC
    >(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where VEC: VecTrait<T> + Copy + Init<T> + Send + Sync, T: IntoScalar<T>
{
    let img_shape = img.shape();
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
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
        img.pad(
            &[
                (0, 0),
                (ph_start, ph_end),
                (pw_start, pw_end),
                (0, 0),
            ],
            T::ZERO
        )?
    } else {
        img.clone()
    };
    let output = _Tensor::<T>::zeros([batch, out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let mut kernel = kernels.ptr();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = output.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels

    let l2_cache = cache_size::l2_cache_size().unwrap_or(256 * 1024) / std::mem::size_of::<T>();
    let cache_line = cache_size::cache_line_size(0, cache_size::CacheType::Data).unwrap_or(64);
    let num_el_in_cache_line = cache_line / std::mem::size_of::<T>();
    let num_cache_set = get_num_cache_set(
        cache_size::l2_cache_size().unwrap_or(256 * 1024),
        cache_line,
        8
    );
    let wo_gap = get_set_gap::<T>(osw as i64, cache_line, num_cache_set);

    // let (co_b, wo_b, ci_b) = find_combination(
    //     l2_cache as i64,
    //     out_channels,
    //     out_width,
    //     in_channels
    // );
    // let (co_b, wo_b, ci_b) = find_exact_combination(
    //     l2_cache as i64,
    //     out_channels as i64,
    //     out_width as i64,
    //     in_channels as i64
    // );
    // println!("co_b: {}, wo_b: {}, ci_b: {}", co_b, wo_b, ci_b);
    let co_b = 80;
    let wo_b = 1024;
    let ci_b = 8;
    // co_b * vec_size * ci_b * 2 + ci_b * wo_b + wo_b * co_b * vec_size * 2 <= l2_cache
    let co_b_remain = co_b % (VECSIZE as i64);
    let co_b = co_b - co_b_remain;

    let num_co_b = out_channels / co_b;
    let num_wo_b = out_width / wo_b;
    let num_ci_b = in_channels / ci_b;

    let wo_b_remain = wo_b % (REGNUM as i64);
    let num_wo_rb = wo_b / (REGNUM as i64);
    let co_b_remain = co_b % (VECSIZE as i64);
    assert_eq!(co_b_remain, 0);
    let num_co_rb = co_b / (VECSIZE as i64);

    let outer = batch * num_co_b * num_ci_b * out_height;
    (0..outer).into_par_iter().for_each_init(
        || { (out, kernel, inp, vec![VEC::splat(T::ZERO); wo_b as usize * num_co_rb as usize]) },
        |(out, kernel, inp, res_vectors), idx| {
            let b = idx / (num_co_b * num_ci_b * out_height);
            let t = (idx / (num_ci_b * out_height)) % num_co_b;
            let ip = (idx / out_height) % num_ci_b;
            let l = idx % out_height;
            out.offset(t * co_b);
            kernel.offset(t * co_b);
            for kp in 0..num_wo_b {
                for k in 0..num_wo_rb {
                    for j in 0..num_co_rb {
                        for p in 0..REGNUM as i64 {
                            let idx = k * (num_co_rb * (REGNUM as i64)) + j * (REGNUM as i64) + p;
                            let k = kp * wo_b + k * (REGNUM as i64) + p;
                            let out_ptr = &mut out[b * osb + l * osh + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
                            unsafe {
                                *res_vectors.get_unchecked_mut(idx as usize) =
                                    VEC::from_ptr(out_ptr);
                            }
                        }
                    }
                }
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for i in 0..ci_b {
                            let i = ip * ci_b + i;
                            for k in 0..num_wo_rb {
                                for j in 0..num_co_rb {
                                    let kernel_vec = unsafe { VEC::from_ptr(&kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j * (VECSIZE as i64)]) }; // prettier-ignore
                                    for p in 0..REGNUM as i64 {
                                        let out_vec = unsafe { res_vectors.get_unchecked_mut((k * (num_co_rb * REGNUM as i64) + j * (REGNUM as i64) + p) as usize) }; // prettier-ignore
                                        let k = kp * wo_b + k * (REGNUM as i64) + p;
                                        let inp_vec = VEC::splat(inp[b * isb + (l * step_height + n * dh) * ish + (k * step_width + m * dw) * isw + ip * ci_b + i]); // prettier-ignore
                                        *out_vec = inp_vec._mul_add(kernel_vec, *out_vec); // prettier-ignore
                                    }
                                }
                            }
                            for j in 0..num_co_rb {
                                for p in 0..wo_b_remain {
                                    let k = kp * wo_b + num_wo_rb * (REGNUM as i64) + p;
                                    for c in 0..VECSIZE {
                                        let j = j * (VECSIZE as i64) + (c as i64);
                                        out[b * osb + l * osh + k * osw + j] +=
                                                    inp[b * isb + (l * step_height + n * dh) * ish + (k * step_width + m * dw) * isw + ip * ci_b + i] *
                                                    kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j]; // prettier-ignore
                                    }
                                }
                            }
                        }
                    }
                }
                for k in 0..num_wo_rb {
                    for j in 0..num_co_rb {
                        for p in 0..REGNUM as i64 {
                            let idx = k * (num_co_rb * (REGNUM as i64)) + j * (REGNUM as i64) + p;
                            let k = kp * wo_b + k * (REGNUM as i64) + p;
                            let out_ptr = &mut out[b * osb + l * osh + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
                            unsafe {
                                std::ptr::copy_nonoverlapping(res_vectors.get_unchecked(idx as usize).as_ptr(), out_ptr, VECSIZE); // prettier-ignore
                            }
                        }
                    }
                }
            }
        }
    ); // end of par_iter

    // for b in 0..batch {
    //     for _ in 0..num_co_b {
    //         for ip in 0..num_ci_b {
    //             for l in 0..out_height {
    //                 for kp in 0..num_wo_b {
    //                     for k in 0..num_wo_rb {
    //                         for j in 0..num_co_rb {
    //                             for p in 0..REGNUM as i64 {
    //                                 let idx = k * (num_co_rb * REGNUM as i64) + j * (REGNUM as i64) + p;
    //                                 let k = kp * wo_b + k * (REGNUM as i64) + p;
    //                                 let out_ptr = &mut out[b * osb + l * osh + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
    //                                 unsafe {
    //                                     *res_vectors.get_unchecked_mut(idx as usize) =
    //                                         VEC::from_ptr(out_ptr);
    //                                 }
    //                             }
    //                         }
    //                     }
    //                     for n in 0..kernel_height {
    //                         for m in 0..kernel_width {
    //                             for i in 0..ci_b {
    //                                 let i = ip * ci_b + i;
    //                                 for k in 0..num_wo_rb {
    //                                     for j in 0..num_co_rb {
    //                                         let kernel_vec = unsafe { VEC::from_ptr(&kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j * (VECSIZE as i64)]) }; // prettier-ignore
    //                                         for p in 0..REGNUM as i64 {
    //                                             let out_vec = unsafe { res_vectors.get_unchecked_mut((k * (num_co_rb * REGNUM as i64) + j * (REGNUM as i64) + p) as usize) }; // prettier-ignore
    //                                             let k = kp * wo_b + k * (REGNUM as i64) + p;
    //                                             let inp_vec = VEC::splat(inp[b * isb + (l * step_height + n * dh) * ish + (k * step_width + m * dw) * isw + ip * ci_b + i]); // prettier-ignore
    //                                             *out_vec = inp_vec._mul_add(kernel_vec, *out_vec); // prettier-ignore
    //                                         }
    //                                     }
    //                                 }
    //                                 for j in 0..num_co_rb {
    //                                     for p in 0..wo_b_remain {
    //                                         let k = kp * wo_b + num_wo_rb * (REGNUM as i64) + p;
    //                                         for c in 0..VECSIZE {
    //                                             let j = j * (VECSIZE as i64) + (c as i64);
    //                                             out[b * osb + l * osh + k * osw + j] +=
    //                                                 inp[b * isb + (l * step_height + n * dh) * ish + (k * step_width + m * dw) * isw + ip * ci_b + i] *
    //                                                 kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j]; // prettier-ignore
    //                                         }
    //                                     }
    //                                 }
    //                             }
    //                         }
    //                     }
    //                     for k in 0..num_wo_rb {
    //                         for j in 0..num_co_rb {
    //                             for p in 0..REGNUM as i64 {
    //                                 let idx = k * (num_co_rb * REGNUM as i64) + j * (REGNUM as i64) + p;
    //                                 let k = kp * wo_b + k * (REGNUM as i64) + p;
    //                                 let out_ptr = &mut out[b * osb + l * osh + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
    //                                 unsafe {
    //                                     std::ptr::copy_nonoverlapping(res_vectors.get_unchecked(idx as usize).as_ptr(), out_ptr, VECSIZE); // prettier-ignore
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         out.offset(co_b);
    //         kernel.offset(co_b);
    //     }
    // }

    Ok(output)
}

pub fn get_num_cache_set(cache_size: usize, cache_line_size: usize, associativity: usize) -> usize {
    cache_size / (cache_line_size * associativity)
}

#[allow(unused)]
pub(crate) fn get_cache_set(
    address: usize,
    cache_line_size: usize,
    num_cache_sets: usize
) -> usize {
    (address / cache_line_size) % num_cache_sets
}

pub(crate) fn get_set_gap<T>(stride: i64, cache_line_size: usize, cache_set_num: usize) -> usize {
    let set1 = get_cache_set(0, cache_line_size, cache_set_num);
    let set2 = get_cache_set(
        ((stride as usize) * std::mem::size_of::<T>()) as usize,
        cache_line_size,
        cache_set_num
    );
    if set2 > set1 {
        set2 - set1
    } else {
        set1 + cache_set_num - set2
    }
}

fn find_combination(max_cache_size: i64, max_x: i64, max_y: i64, max_z: i64) -> (i64, i64, i64) {
    let mut left = 1;
    let mut right = max_x;
    let mut best_x = 0;
    let mut best_y = 0;
    let mut best_z = 0;

    while left <= right {
        let mid_x = left + (right - left) / 2;
        let x_size = mid_x;

        if x_size > max_cache_size {
            right = mid_x - 1;
        } else {
            let remaining_cache_after_x = max_cache_size - x_size;
            let y = max_y.min(remaining_cache_after_x);
            let y_size = y;

            let remaining_cache_after_xy = remaining_cache_after_x - y_size;
            let z = max_z.min(remaining_cache_after_xy);

            // 如果合法，更新 best_x, best_y, best_z
            if mid_x * y * z <= max_cache_size {
                best_x = mid_x;
                best_y = y;
                best_z = z;
                left = mid_x + 1; // 尝试更大的 x
            } else {
                right = mid_x - 1; // 尝试更小的 x
            }
        }
    }

    (best_x, best_y, best_z)
}

fn find_exact_combination(
    max_cache_size: i64,
    max_co_b: i64,
    max_wo_b: i64,
    max_ci_b: i64
) -> (i64, i64, i64) {
    let mut best_co_b = 0;
    let mut best_wo_b = 0;
    let mut best_ci_b = 0;

    for co_b in (1..max_co_b + 1).rev() {
        for wo_b in (1..max_wo_b + 1).rev() {
            for ci_b in (1..max_ci_b + 1).rev() {
                let product = co_b * ci_b + ci_b * wo_b + wo_b * co_b;
                if product <= max_cache_size {
                    if
                        co_b > best_co_b ||
                        (co_b == best_co_b && wo_b > best_wo_b) ||
                        (co_b == best_co_b && wo_b == best_wo_b && ci_b > best_ci_b)
                    {
                        best_co_b = co_b;
                        best_wo_b = wo_b;
                        best_ci_b = ci_b;
                    }
                }
            }
        }
    }

    (best_co_b, best_wo_b, best_ci_b)
}
