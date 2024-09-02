use aligned_vec::AVec;
use aligned_vec::ConstAlign;
use tensor_common::pointer::Pointer;
use tensor_types::vectors::traits::*;
use crate::tensor_base::_Tensor;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use rayon::prelude::*;

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
    use std::{ arch::x86_64::_mm_prefetch, sync::Arc };

    use aligned_vec::avec;
    use tensor_common::shape_utils::mt_intervals;

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
    let output = _Tensor::<T>::empty([batch, out_height, out_width, out_channels])?;
    let out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = output.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels

    let l1_cache = cache_size::l1_cache_size().unwrap_or(32 * 1024) / std::mem::size_of::<T>();
    let l2_cache = cache_size::l2_cache_size().unwrap_or(256 * 1024) / std::mem::size_of::<T>();
    let cache_line = cache_size::cache_line_size(0, cache_size::CacheType::Data).unwrap_or(64);
    let num_cache_set = get_num_cache_set(
        cache_size::l2_cache_size().unwrap_or(256 * 1024),
        cache_line,
        8
    );
    // let (co_b, wo_b, ci_b) = find_exact_combination(
    //     l2_cache as i64,
    //     out_channels as i64,
    //     out_width as i64,
    //     in_channels as i64
    // );
    let co_b = 80;
    let wo_b = 1024;
    let ci_b = 8;
    // println!(
    //     "co_b: {}, wo_b: {}, ci_b: {}, cache: {}, {}",
    //     co_b,
    //     wo_b,
    //     ci_b,
    //     l1_cache,
    //     co_b * wo_b * ci_b
    // );
    // co_b * vec_size * ci_b * 2 + ci_b * wo_b + wo_b * co_b * vec_size * 2 <= l2_cache
    let co_b_remain = co_b % (VECSIZE as i64);
    let co_b = co_b - co_b_remain;

    let num_co_b = out_channels / co_b;
    let num_wo_b = out_width / wo_b;
    let num_ci_b = in_channels / ci_b;
    // println!("num_co_b: {}, num_wo_b: {}, num_ci_b: {}", num_co_b, num_wo_b, num_ci_b);

    let wo_b_remain = wo_b % (REGNUM as i64); /* 3 */
    let num_wo_rb = wo_b / (REGNUM as i64); /* 6 */
    let num_co_rb = co_b / (VECSIZE as i64); /* 8 */

    // println!("num_wo_rb: {}, num_co_rb: {}", num_wo_rb, num_co_rb);
    // println!("wo_b_remain: {}", wo_b_remain);
    // println!("co_b_remain: {}", co_b_remain);
    // let res_vectors =
    //     vec![[VEC::splat(T::ZERO); REGNUM]; num_wo_rb as usize * num_co_rb as usize]; /* 11760 elements */ /* L1 = 12288 elements */
    // let inp_vectors = avec![T::ZERO; wo_b as usize * ci_b as usize];
    // let kernel_vectors = vec![VEC::splat(T::ZERO); num_co_rb as usize];
    // let remain_vectors = vec![VEC::splat(T::ZERO); wo_b_remain as usize * num_co_rb as usize];
    // println!(
    //     "buffers: {}",
    //     res_vectors.len() * REGNUM * VECSIZE +
    //         REGNUM * (ci_b as usize) * (wo_b as usize) +
    //         (num_co_rb as usize) * VECSIZE +
    //         (wo_b_remain as usize) * (num_co_rb as usize) * VECSIZE
    // );
    // println!(
    //     "buffer len: {}",
    //     res_vectors.len() + inp_vectors.len() + kernel_vectors.len() + remain_vectors.len()
    // );
    let outer = batch * num_co_b * num_ci_b * out_height;
    let intervals = Arc::new(mt_intervals(outer as usize, rayon::current_num_threads()));
    (0..rayon::current_num_threads()).into_par_iter().for_each_init(
        || { (out, kernel, inp) },
        |(out, kernel, inp), idx| {
            let (start, end) = intervals[idx];
            let mut res_vectors =
                avec![[VEC::splat(T::ZERO); REGNUM]; num_wo_rb as usize * num_co_rb as usize];
            let mut remain_vectors =
                avec![VEC::splat(T::ZERO); wo_b_remain as usize * num_co_rb as usize];
            let mut kernel_vectors = avec![VEC::splat(T::ZERO); num_co_rb as usize * ci_b as usize];
            let mut inp_vectors = avec![T::ZERO; wo_b as usize * ci_b as usize];
            for idx in start as i64..end as i64 {
                let b = idx / (num_co_b * num_ci_b * out_height);
                let t = (idx / (num_ci_b * out_height)) % num_co_b;
                let ip = (idx / out_height) % num_ci_b;
                let l = idx % out_height;
                out.offset(t * co_b);
                kernel.offset(t * co_b);
                for kp in 0..num_wo_b {
                    pack_res::<T, VEC, REGNUM, VECSIZE>(
                        num_wo_rb,
                        num_co_rb,
                        osw,
                        b * osb + l * osh + kp * wo_b * osw,
                        out,
                        &mut res_vectors
                    );
                    pack_remain_res::<T, VEC, REGNUM, VECSIZE>(
                        num_co_rb,
                        wo_b_remain,
                        osw,
                        kp * wo_b + num_wo_rb * (REGNUM as i64),
                        b * osb + l * osh,
                        out,
                        &mut remain_vectors
                    );
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            pack_input::<T, VEC, REGNUM, VECSIZE>(
                                ci_b,
                                num_wo_rb,
                                b * isb +
                                    (l * step_height + n * dh) * ish +
                                    kp * wo_b * isw +
                                    m * dw * isw +
                                    ip * ci_b,
                                step_width,
                                isw,
                                inp,
                                &mut inp_vectors
                            );
                            pack_kernel::<T, VEC, REGNUM, VECSIZE>(
                                ci_b,
                                ip,
                                num_co_rb,
                                n * ks0 + m * ks1,
                                ks2,
                                kernel,
                                &mut kernel_vectors
                            );
                            for ii in 0..ci_b {
                                let i = ip * ci_b + ii;
                                for k in 0..num_wo_rb {
                                    let offset =
                                        ii * num_wo_rb * (REGNUM as i64) + k * (REGNUM as i64);
                                    let inp_vecs = get_inp_vectors::<T, VEC, REGNUM>(
                                        offset as usize,
                                        &inp_vectors
                                    );
                                    do_calculate::<T, VEC, REGNUM, VECSIZE>(
                                        num_co_rb,
                                        k,
                                        i,
                                        inp_vecs,
                                        &mut res_vectors,
                                        &kernel_vectors
                                    );
                                }
                                let _k = kp * wo_b + num_co_rb * (REGNUM as i64) + 0;
                                let inp_vec0 = VEC::splat(inp[b * isb + (l * step_height + n * dh) * ish + (_k * step_width + m * dw) * isw + ip * ci_b + i]); // prettier-ignore

                                let _k = kp * wo_b + num_co_rb * (REGNUM as i64) + 1;
                                let inp_vec1 = VEC::splat(inp[b * isb + (l * step_height + n * dh) * ish + (_k * step_width + m * dw) * isw + ip * ci_b + i]); // prettier-ignore
                                for j in 0..num_co_rb {
                                    let idx = j * wo_b_remain + 0;
                                    let kernel_vec = unsafe { VEC::from_ptr(&kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j * (VECSIZE as i64)]) }; // prettier-ignore

                                    let out_vec = unsafe { remain_vectors.get_unchecked_mut(idx as usize) }; // prettier-ignore
                                    *out_vec = inp_vec0._mul_add(kernel_vec, *out_vec); // prettier-ignore

                                    let idx = j * wo_b_remain + 1;
                                    let out_vec = unsafe { remain_vectors.get_unchecked_mut(idx as usize) }; // prettier-ignore
                                    *out_vec = inp_vec1._mul_add(kernel_vec, *out_vec); // prettier-ignore
                                }
                            }
                        }
                    }
                    store_res::<T, VEC, REGNUM, VECSIZE>(
                        num_wo_rb,
                        num_co_rb,
                        kp * wo_b,
                        osw,
                        b * osb + l * osh,
                        out,
                        &mut res_vectors
                    );
                    store_remain_res::<T, VEC, REGNUM, VECSIZE>(
                        num_wo_rb,
                        num_co_rb,
                        wo_b_remain,
                        kp * wo_b,
                        osw,
                        b * osb + l * osh,
                        out,
                        &mut remain_vectors
                    );
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
    //                             let idx = k * num_co_rb + j;
    //                             let vectors = unsafe {
    //                                 res_vectors.get_unchecked_mut(idx as usize)
    //                             };
    //                             for p in 0..REGNUM as i64 {
    //                                 let k = kp * wo_b + k * (REGNUM as i64) + p;
    //                                 let out_ptr = &mut out[b * osb + l * osh + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
    //                                 let vector = &mut vectors[p as usize];
    //                                 unsafe {
    //                                     *vector = VEC::from_ptr(out_ptr);
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
    //                                         let idx = k * num_co_rb + j;
    //                                         let kernel_vec = unsafe { VEC::from_ptr(&kernel[n * ks0 + m * ks1 + (ip * ci_b + i) * ks2 + j * (VECSIZE as i64)]) }; // prettier-ignore
    //                                         for p in 0..REGNUM as i64 {
    //                                             let out_vec = unsafe { res_vectors.get_unchecked_mut(idx as usize) }; // prettier-ignore
    //                                             let k = kp * wo_b + k * (REGNUM as i64) + p;
    //                                             let inp_vec = VEC::splat(inp[b * isb + (l * step_height + n * dh) * ish + (k * step_width + m * dw) * isw + ip * ci_b + i]); // prettier-ignore
    //                                             out_vec[p as usize] = inp_vec._mul_add(kernel_vec, out_vec[p as usize]); // prettier-ignore
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
    //                             let idx = k * num_co_rb + j;
    //                             let res_vectors = unsafe {
    //                                 res_vectors.get_unchecked_mut(idx as usize)
    //                             };
    //                             for p in 0..REGNUM as i64 {
    //                                 let k = kp * wo_b + k * (REGNUM as i64) + p;
    //                                 let out_ptr = &mut out[b * osb + l * osh + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
    //                                 unsafe {
    //                                     std::ptr::copy_nonoverlapping(res_vectors[p as usize].as_ptr(), out_ptr, VECSIZE); // prettier-ignore
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
#[allow(unused)]
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

#[allow(unused)]
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

fn pack_res<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    num_wo_rb: i64,
    num_co_rb: i64,
    osw: i64,
    offset: i64,
    out: &mut Pointer<T>,
    res_vectors: &mut aligned_vec::AVec<[VEC; REGNUM], aligned_vec::ConstAlign<128>>
)
    where T: CommonBounds, VEC: Init<T>
{
    for k in 0..num_wo_rb {
        for j in 0..num_co_rb {
            let idx = k * num_co_rb + j;
            let offset = offset + k * (REGNUM as i64) * osw + j * (VECSIZE as i64);
            unsafe {
                let vectors = res_vectors.get_unchecked_mut(idx as usize);

                let out_ptr0 = & out[offset + 0 * osw]; // prettier-ignore
                let vector0 = vectors.get_unchecked_mut(0) as *mut VEC;

                let out_ptr1 = & out[offset + 1 * osw]; // prettier-ignore
                let vector1 = vectors.get_unchecked_mut(1) as *mut VEC;

                let out_ptr2 = &out[offset + 2 * osw]; // prettier-ignore
                let vector2 = vectors.get_unchecked_mut(2) as *mut VEC;

                let out_ptr3 = &out[offset + 3 * osw]; // prettier-ignore
                let vector3 = vectors.get_unchecked_mut(3) as *mut VEC;

                let out_ptr4 = &out[offset + 4 * osw]; // prettier-ignore
                let vector4 = vectors.get_unchecked_mut(4) as *mut VEC;

                let out_ptr5 = &out[offset]; // prettier-ignore
                let vector5 = vectors.get_unchecked_mut(5) as *mut VEC;

                let out_ptr6 = &out[offset + 5 * osw]; // prettier-ignore
                let vector6 = vectors.get_unchecked_mut(6) as *mut VEC;

                *vector0 = VEC::from_ptr(out_ptr0);
                *vector1 = VEC::from_ptr(out_ptr1);
                *vector2 = VEC::from_ptr(out_ptr2);
                *vector3 = VEC::from_ptr(out_ptr3);
                *vector4 = VEC::from_ptr(out_ptr4);
                *vector5 = VEC::from_ptr(out_ptr5);
                *vector6 = VEC::from_ptr(out_ptr6);
            }
        }
    }
}

fn pack_remain_res<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    num_co_rb: i64,
    wo_b_remain: i64,
    osw: i64,
    k_offset: i64,
    offset: i64,
    out: &mut Pointer<T>,
    remain_vectors: &mut AVec<VEC, ConstAlign<128>>
)
    where T: CommonBounds, VEC: Init<T>
{
    for j in 0..num_co_rb {
        for p in 0..wo_b_remain {
            let idx = j * wo_b_remain + p;
            let k = k_offset + p;
            let out_ptr = &mut out[offset + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
            unsafe {
                *remain_vectors.get_unchecked_mut(idx as usize) = VEC::from_ptr(out_ptr);
            }
        }
    }
}

fn pack_input<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    ci_b: i64,
    num_wo_rb: i64,
    offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    inp_vectors: &mut AVec<T, ConstAlign<128>>
)
    where T: CommonBounds
{
    for i in 0..ci_b {
        for k in 0..num_wo_rb {
            let offset = offset + k * (REGNUM as i64) * step_width * isw + i; // prettier-ignore
            for r in 0..REGNUM as i64 {
                let idx = i * num_wo_rb * (REGNUM as i64) + k * (REGNUM as i64) + r;
                let inp_vec = unsafe { inp_vectors.get_unchecked_mut(idx as usize) };
                *inp_vec = inp[offset + (r as i64) * step_width * isw];
            }
        }
    }
}

fn pack_kernel<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    ci_b: i64,
    ip: i64,
    num_co_rb: i64,
    offset: i64,
    ks2: i64,
    kernel: &Pointer<T>,
    kernel_vectors: &mut AVec<VEC, ConstAlign<128>>
)
    where T: CommonBounds, VEC: Init<T>
{
    for i in 0..ci_b {
        let i = ip * ci_b + i;
        for j in 0..num_co_rb {
            let idx = i * num_co_rb + j;
            let kernel_vec = unsafe { VEC::from_ptr(&kernel[offset + (ip * ci_b + i) * ks2 + j * (VECSIZE as i64)]) }; // prettier-ignore
            *(unsafe { kernel_vectors.get_unchecked_mut(idx as usize) }) = kernel_vec;
        }
    }
}

fn get_inp_vectors<T, VEC, const REGNUM: usize>(
    offset: usize,
    inp_vectors: &AVec<T, ConstAlign<128>>
)
    -> [VEC; REGNUM]
    where T: CommonBounds, VEC: Init<T>
{
    array_init::array_init(|idx| {
        VEC::splat(unsafe { *inp_vectors.get_unchecked(offset + idx) })
    })
}

fn do_calculate<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    num_co_rb: i64,
    k: i64,
    i: i64,
    inp_vec: [VEC; REGNUM],
    res_vectors: &mut aligned_vec::AVec<[VEC; REGNUM], aligned_vec::ConstAlign<128>>,
    kernel_vectors: &AVec<VEC, ConstAlign<128>>
)
    where T: CommonBounds, VEC: VecTrait<T> + Copy
{
    assert_eq!(inp_vec.len(), 7);
    for j in 0..num_co_rb {
        let idx = k * num_co_rb + j;
        let kernel_idx = i * num_co_rb + j;
        let res_vectors = unsafe { res_vectors.get_unchecked_mut(idx as usize) };
        let kernel_vec = *(unsafe { kernel_vectors.get_unchecked(kernel_idx as usize) });
        let out_vec0 = unsafe { res_vectors.get_unchecked_mut(0) as *mut VEC }; // prettier-ignore
        let out_vec1 = unsafe { res_vectors.get_unchecked_mut(1) as *mut VEC }; // prettier-ignore
        let out_vec2 = unsafe { res_vectors.get_unchecked_mut(2) as *mut VEC }; // prettier-ignore
        let out_vec3 = unsafe { res_vectors.get_unchecked_mut(3) as *mut VEC }; // prettier-ignore
        let out_vec4 = unsafe { res_vectors.get_unchecked_mut(4) as *mut VEC }; // prettier-ignore
        let out_vec5 = unsafe { res_vectors.get_unchecked_mut(5) as *mut VEC }; // prettier-ignore
        let out_vec6 = unsafe { res_vectors.get_unchecked_mut(6) as *mut VEC }; // prettier-ignore

        let res0 = inp_vec[0]._mul_add(kernel_vec, unsafe { out_vec0.read() }); // prettier-ignore
        let res1 = inp_vec[1]._mul_add(kernel_vec, unsafe { out_vec1.read() }); // prettier-ignore
        let res2 = inp_vec[2]._mul_add(kernel_vec, unsafe { out_vec2.read() }); // prettier-ignore
        let res3 = inp_vec[3]._mul_add(kernel_vec, unsafe { out_vec3.read() }); // prettier-ignore
        let res4 = inp_vec[4]._mul_add(kernel_vec, unsafe { out_vec4.read() }); // prettier-ignore
        let res5 = inp_vec[5]._mul_add(kernel_vec, unsafe { out_vec5.read() }); // prettier-ignore
        let res6 = inp_vec[6]._mul_add(kernel_vec, unsafe { out_vec6.read() }); // prettier-ignore

        unsafe {
            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
            *out_vec4 = res4;
            *out_vec5 = res5;
            *out_vec6 = res6;
        }
    }
}

fn store_res<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    num_wo_rb: i64,
    num_co_rb: i64,
    k_offset: i64,
    osw: i64,
    offset: i64,
    out: &mut Pointer<T>,
    res_vectors: &mut aligned_vec::AVec<[VEC; REGNUM], aligned_vec::ConstAlign<128>>
)
    where T: CommonBounds, VEC: VecTrait<T> + Copy
{
    for k in 0..num_wo_rb {
        for j in 0..num_co_rb {
            let idx = k * num_co_rb + j;
            let res_vectors = unsafe { res_vectors.get_unchecked_mut(idx as usize) };
            for p in 0..REGNUM as i64 {
                let k = k_offset + k * (REGNUM as i64) + p;
                let out_ptr = &mut out[offset + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
                unsafe {
                    std::ptr::copy_nonoverlapping(res_vectors[p as usize].as_ptr(), out_ptr, VECSIZE); // prettier-ignore
                }
            }
        }
    }
}

fn store_remain_res<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    num_wo_rb: i64,
    num_co_rb: i64,
    wo_b_remain: i64,
    k_offset: i64,
    osw: i64,
    offset: i64,
    out: &mut Pointer<T>,
    remain_vectors: &mut AVec<VEC, ConstAlign<128>>
)
    where T: CommonBounds, VEC: VecTrait<T> + Copy
{
    for j in 0..num_co_rb {
        for p in 0..wo_b_remain {
            let idx = j * wo_b_remain + p;
            let k = k_offset + num_wo_rb * (REGNUM as i64) + p;
            let out_ptr = &mut out[offset + k * osw + j * (VECSIZE as i64)]; // prettier-ignore
            unsafe {
                std::ptr::copy_nonoverlapping(remain_vectors.get_unchecked(idx as usize).as_ptr(), out_ptr, VECSIZE); // prettier-ignore
            }
        }
    }
}
