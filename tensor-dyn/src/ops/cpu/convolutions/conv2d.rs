use tensor_common::pointer::Pointer;
use tensor_types::vectors::traits::*;
use crate::tensor_base::_Tensor;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use rayon::prelude::*;
use num::traits::MulAdd;
use tensor_types::dtype::TypeCommon;

impl<T> _Tensor<T>
    where
        T: CommonBounds +
            std::ops::Mul<Output = T> +
            std::ops::AddAssign +
            MulAdd<Output = T> +
            IntoScalar<T>,
        <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + Send + Sync + VecSize
{
    pub fn conv2d(
        &self,
        kernels: &_Tensor<T>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2]
    ) -> anyhow::Result<_Tensor<T>> {
        use tensor_common::shape_utils::mt_intervals;

        use crate::CONV_REGNUM;

        let img_shape = self.shape();
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
        let out_width =
            (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
        let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
            self.pad(
                &[
                    (0, 0),
                    (ph_start, ph_end),
                    (pw_start, pw_end),
                    (0, 0),
                ],
                T::ZERO
            )?
        } else {
            self.clone()
        };
        let output = _Tensor::<T>::zeros([batch, out_height, out_width, out_channels])?;
        let out = output.ptr();
        let inp = img.ptr();
        let kernel = kernels.ptr();

        let osb = output.strides()[0]; // batch
        let osh = output.strides()[1]; // height
        let osw = output.strides()[2]; // width

        let isb = img.strides()[0]; // batch
        let ish = img.strides()[1]; // height
        let isw = img.strides()[2]; // width

        let ks0 = kernels.strides()[0]; // kernel_height
        let ks1 = kernels.strides()[1]; // kernel_width
        let ks2 = kernels.strides()[2]; // in_channels

        let l1_cache =
            cache_size::l1_cache_size().unwrap_or(32 * 1024 /* 32 kb */) / std::mem::size_of::<T>();

        let (co_b, ci_b) = find_exact_combination::<T, CONV_REGNUM>(
            l1_cache as i64,
            out_channels as i64,
            in_channels as i64,
            kernel_height as i64,
            kernel_width as i64
        );
        let num_co_b = out_channels / co_b;
        let num_wo_b = out_width / (CONV_REGNUM as i64);
        let num_ci_b = in_channels / ci_b;

        let co_b_remain = out_channels % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        let wo_b_remain = out_width % (CONV_REGNUM as i64);
        let ci_b_remain = in_channels % ci_b;
        let num_co_rb = co_b / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);

        let outer = batch * num_co_b * num_ci_b * out_height;

        let case0 = move |b: i64, l: i64, c: i64, ip: i64, ci_b_remain: i64, mut out: Pointer<T>| {
            for kp in 0..num_wo_b {
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for ii in 0..ci_b_remain {
                            let i = ip * ci_b + ii;
                            micro_kernel_regnum::<T, CONV_REGNUM>(
                                num_co_rb,
                                kp,
                                i,
                                b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                                c * co_b,
                                b * osb + l * osh + kp * CONV_REGNUM as i64 * osw, // prettier-ignore
                                n * ks0 + m * ks1 + i * ks2,
                                step_width,
                                isw,
                                osw,
                                &inp,
                                &mut out,
                                &kernel
                            );
                        }
                    }
                }
            }
        };

        let case1_helper = move |
            b: i64,
            l: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            micro_kernel: fn(
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                &Pointer<T>,
                &mut Pointer<T>,
                &Pointer<T>
            ),
            mut out: Pointer<T>
        | {
            for n in 0..kernel_height {
                for m in 0..kernel_width {
                    for ii in 0..ci_b_remain {
                        let i = ip * ci_b + ii;
                        micro_kernel(
                            num_co_rb,
                            num_wo_b,
                            i,
                            b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                            c * co_b,
                            b * osb + l * osh + num_wo_b * CONV_REGNUM as i64 * osw, // prettier-ignore
                            n * ks0 + m * ks1 + i * ks2,
                            step_width,
                            isw,
                            osw,
                            &inp,
                            &mut out,
                            &kernel
                        );
                    }
                }
            }
        };

        let case1 = move |b: i64, l: i64, c: i64, ip: i64, ci_b_remain: i64, out: Pointer<T>| {
            match wo_b_remain {
                1 => {
                    case1_helper(b, l, c, ip, ci_b_remain, micro_kernel_1::<T, CONV_REGNUM>, out);
                }
                2 => {
                    case1_helper(b, l, c, ip, ci_b_remain, micro_kernel_2::<T, CONV_REGNUM>, out);
                }
                3 => {
                    case1_helper(b, l, c, ip, ci_b_remain, micro_kernel_3::<T, CONV_REGNUM>, out);
                }
                4 => {
                    case1_helper(b, l, c, ip, ci_b_remain, micro_kernel_4::<T, CONV_REGNUM>, out);
                }
                5 => {
                    case1_helper(b, l, c, ip, ci_b_remain, micro_kernel_5::<T, CONV_REGNUM>, out);
                }
                6 => {
                    case1_helper(b, l, c, ip, ci_b_remain, micro_kernel_6::<T, CONV_REGNUM>, out);
                }
                _ => unimplemented!(),
            }
        };

        let case2 = move |b: i64, l: i64, c: i64, ip: i64, ci_b_remain: i64, mut out: Pointer<T>| {
            let mut res_buffer =
                vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); CONV_REGNUM]; num_co_rb as usize + 1];
            for kp in 0..num_wo_b {
                load_store_res_buffer::<T, CONV_REGNUM, true>(
                    num_co_rb,
                    co_b_remain,
                    osw,
                    c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                    &mut res_buffer,
                    &mut out
                );
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for ii in 0..ci_b_remain {
                            let i = ip * ci_b + ii;
                            // need either packing or minimize the cost of cache miss
                            micro_kernel_regnum_with_buffer::<T, CONV_REGNUM>(
                                num_co_rb,
                                kp,
                                i,
                                b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                                c * co_b,
                                n * ks0 + m * ks1 + i * ks2,
                                step_width,
                                isw,
                                &inp,
                                &mut res_buffer,
                                &kernel
                            );
                        }
                    }
                }
                load_store_res_buffer::<T, CONV_REGNUM, false>(
                    num_co_rb,
                    co_b_remain,
                    osw,
                    c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                    &mut res_buffer,
                    &mut out
                );
            }
        };

        let case3_helper = move |
            b: i64,
            l: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            wo_b_remain: i64,
            pack_kernel: fn(i64, i64, i64, &Pointer<T>, &mut Vec<<T as TypeCommon>::Vec>),
            load_fn: fn(i64, i64, i64, i64, &mut Vec<Vec<<T as TypeCommon>::Vec>>, &mut Pointer<T>),
            store_fn: fn(
                i64,
                i64,
                i64,
                i64,
                &mut Vec<Vec<<T as TypeCommon>::Vec>>,
                &mut Pointer<T>
            ),
            micro_kernel: fn(
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                &Pointer<T>,
                &mut Vec<Vec<<T as TypeCommon>::Vec>>,
                &[<T as TypeCommon>::Vec]
            ),
            mut out: Pointer<T>
        | {
            let mut kernel_buffer =
                vec![<T as TypeCommon>::Vec::splat(T::ZERO); num_co_rb as usize + 1];
            let mut remain_buffer =
                vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); wo_b_remain as usize]; num_co_rb as usize + 1];
            for kp in num_wo_b..num_wo_b + 1 {
                load_fn(
                    num_co_rb,
                    co_b_remain,
                    osw,
                    c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                    &mut remain_buffer,
                    &mut out
                );
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for ii in 0..ci_b_remain {
                            let i = ip * ci_b + ii;
                            pack_kernel(
                                num_co_rb,
                                co_b_remain,
                                c * co_b + n * ks0 + m * ks1 + i * ks2,
                                &kernel,
                                &mut kernel_buffer
                            );
                            micro_kernel(
                                num_co_rb,
                                kp,
                                i,
                                b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                                step_width,
                                isw,
                                &inp,
                                &mut remain_buffer,
                                &kernel_buffer
                            );
                        }
                    }
                }
                store_fn(
                    num_co_rb,
                    co_b_remain,
                    osw,
                    c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                    &mut remain_buffer,
                    &mut out
                );
            }
        };

        let case3 = move |b: i64, l: i64, c: i64, ip: i64, ci_b_remain: i64, out: Pointer<T>| {
            match wo_b_remain {
                1 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        1,
                        pack_kernel::<T>,
                        load_store_res_buffer::<T, 1, true>,
                        load_store_res_buffer::<T, 1, false>,
                        micro_kernel_1_with_buffer::<T, CONV_REGNUM, 1>,
                        out
                    );
                }
                2 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        2,
                        pack_kernel::<T>,
                        load_store_res_buffer::<T, 2, true>,
                        load_store_res_buffer::<T, 2, false>,
                        micro_kernel_2_with_buffer::<T, CONV_REGNUM, 2>,
                        out
                    );
                }
                3 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        3,
                        pack_kernel::<T>,
                        load_store_res_buffer::<T, 3, true>,
                        load_store_res_buffer::<T, 3, false>,
                        micro_kernel_3_with_buffer::<T, CONV_REGNUM, 3>,
                        out
                    );
                }
                4 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        4,
                        pack_kernel::<T>,
                        load_store_res_buffer::<T, 4, true>,
                        load_store_res_buffer::<T, 4, false>,
                        micro_kernel_4_with_buffer::<T, CONV_REGNUM, 4>,
                        out
                    );
                }
                5 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        5,
                        pack_kernel::<T>,
                        load_store_res_buffer::<T, 5, true>,
                        load_store_res_buffer::<T, 5, false>,
                        micro_kernel_5_with_buffer::<T, CONV_REGNUM, 5>,
                        out
                    );
                }
                6 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        6,
                        pack_kernel::<T>,
                        load_store_res_buffer::<T, 6, true>,
                        load_store_res_buffer::<T, 6, false>,
                        micro_kernel_6_with_buffer::<T, CONV_REGNUM, 6>,
                        out
                    );
                }
                _ => unimplemented!(),
            }
        };

        match (co_b_remain == 0, wo_b_remain == 0, ci_b_remain == 0) {
            (true, true, true) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    case0(b, l, c, ip, ci_b, out);
                });
            }
            (true, false, true) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    case0(b, l, c, ip, ci_b, out);
                    case1(b, l, c, ip, ci_b, out);
                });
            }
            (false, true, true) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    if c < num_co_b - 1 {
                        case0(b, l, c, ip, ci_b, out);
                    } else {
                        case2(b, l, c, ip, ci_b, out);
                    }
                });
            }
            (false, false, true) => {
                let intervals = mt_intervals(outer as usize, outer as usize);
                intervals.into_par_iter().for_each(|(start, end)| {
                    for idx in start as i64..end as i64 {
                        let b = idx / (num_co_b * num_ci_b * out_height);
                        let c = (idx / (num_ci_b * out_height)) % num_co_b;
                        let ip = (idx / out_height) % num_ci_b;
                        let l = idx % out_height;
                        if c < num_co_b - 1 {
                            case0(b, l, c, ip, ci_b, out);
                            case1(b, l, c, ip, ci_b, out);
                        } else {
                            case2(b, l, c, ip, ci_b, out);
                            case3(b, l, c, ip, ci_b, out);
                        }
                    }
                });
            }
            (true, true, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        case0(b, l, c, ip, ci_b, out);
                    } else {
                        // when ip == num_ci_b - 1, we first need to process not remain part, then remain part
                        case0(b, l, c, ip, ci_b, out);
                        case0(b, l, c, in_channels / ci_b, ci_b_remain, out);
                    }
                });
            }
            (true, false, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        case0(b, l, c, ip, ci_b, out);
                        case1(b, l, c, ip, ci_b, out);
                    } else {
                        // when ip == num_ci_b - 1, we first need to process not remain part, then remain part
                        case0(b, l, c, ip, ci_b, out);
                        case1(b, l, c, ip, ci_b, out);
                        case0(b, l, c, in_channels / ci_b, ci_b_remain, out);
                        case1(b, l, c, in_channels / ci_b, ci_b_remain, out);
                    }
                });
            }
            (false, true, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        if c < num_co_b - 1 {
                            case0(b, l, c, ip, ci_b, out);
                        } else {
                            case2(b, l, c, ip, ci_b, out);
                        }
                    } else {
                        // when ip == num_ci_b - 1, we first need to process not remain part, then remain part
                        if c < num_co_b - 1 {
                            case0(b, l, c, ip, ci_b, out);
                            case0(b, l, c, in_channels / ci_b, ci_b_remain, out);
                        } else {
                            case2(b, l, c, ip, ci_b, out);
                            case2(b, l, c, in_channels / ci_b, ci_b_remain, out);
                        }
                    }
                });
            }
            (false, false, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_height);
                    let c = (idx / (num_ci_b * out_height)) % num_co_b;
                    let ip = (idx / out_height) % num_ci_b;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        if c < num_co_b - 1 {
                            case0(b, l, c, ip, ci_b, out);
                            case1(b, l, c, ip, ci_b, out);
                        } else {
                            case2(b, l, c, ip, ci_b, out);
                            case3(b, l, c, ip, ci_b, out);
                        }
                    } else {
                        if c < num_co_b - 1 {
                            case0(b, l, c, ip, ci_b, out);
                            case1(b, l, c, ip, ci_b, out);
                            case0(b, l, c, in_channels / ci_b, ci_b_remain, out);
                            case1(b, l, c, in_channels / ci_b, ci_b_remain, out);
                        } else {
                            case2(b, l, c, ip, ci_b, out);
                            case3(b, l, c, ip, ci_b, out);
                            case2(b, l, c, in_channels / ci_b, ci_b_remain, out);
                            case3(b, l, c, in_channels / ci_b, ci_b_remain, out);
                        }
                    }
                });
            }
        }
        Ok(output)
    }
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

fn find_exact_combination<T: CommonBounds, const REGNUM: usize>(
    max_cache_size: i64,
    max_co_b: i64,
    max_ci_b: i64,
    weight_size: i64,
    height_size: i64
) -> (i64, i64)
    where <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let mut best_co_b = 0;
    let mut best_ci_b = 0;

    for co_b in (1..=max_co_b)
        .rev()
        .filter(|&co_b| co_b % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) == 0) {
        // 只遍历 wo_b 是 7 的倍数的情况
        for ci_b in (1..=max_ci_b).rev() {
            let product =
                co_b * (REGNUM as i64) +
                weight_size * height_size * ci_b * ((REGNUM as i64) + co_b);

            if product <= max_cache_size {
                if co_b > best_co_b || (co_b == best_co_b && ci_b > best_ci_b) {
                    best_co_b = co_b;
                    best_ci_b = ci_b;
                }
            }
        }
    }

    (best_co_b, best_ci_b)
}

fn micro_kernel_regnum<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 6;
    let inp_vec6 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel
                    [
                        kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)
                    ] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec4 = &mut out[co_offset + ofs + 4 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec5 = &mut out[co_offset + ofs + 5 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec6 = &mut out[co_offset + ofs + 6 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());
            let res3 = inp_vec3._mul_add(kernel_vec, out_vec3.read_unaligned());
            let res4 = inp_vec4._mul_add(kernel_vec, out_vec4.read_unaligned());
            let res5 = inp_vec5._mul_add(kernel_vec, out_vec5.read_unaligned());
            let res6 = inp_vec6._mul_add(kernel_vec, out_vec6.read_unaligned());

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
            out_vec3.write_unaligned(res3);
            out_vec4.write_unaligned(res4);
            out_vec5.write_unaligned(res5);
            out_vec6.write_unaligned(res6);
        }
    }
}

fn micro_kernel_regnum_with_buffer<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let inp: &Pointer<T> = &inp;
    let kernel: &Pointer<T> = &kernel;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 6;
    let inp_vec6 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(&kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _); // prettier-ignore
            let res_vectors = res_buffer.get_unchecked_mut(j as usize); // prettier-ignore

            let out_vec0 = res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec3 = res_vectors.get_unchecked_mut(3) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec4 = res_vectors.get_unchecked_mut(4) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec5 = res_vectors.get_unchecked_mut(5) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec6 = res_vectors.get_unchecked_mut(6) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            // perform fused mul add operation
            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);
            let res1 = inp_vec1._mul_add(kernel_vec, *out_vec1);
            let res2 = inp_vec2._mul_add(kernel_vec, *out_vec2);
            let res3 = inp_vec3._mul_add(kernel_vec, *out_vec3);
            let res4 = inp_vec4._mul_add(kernel_vec, *out_vec4);
            let res5 = inp_vec5._mul_add(kernel_vec, *out_vec5);
            let res6 = inp_vec6._mul_add(kernel_vec, *out_vec6);

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

fn micro_kernel_1<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            ); // prettier-ignore

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec0 as *const _)
            );

            out_vec0.write_unaligned(res0);
        }
    }
}

fn micro_kernel_2<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel
                    [

                            co_offset +
                            kernel_offset +
                            j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)

                    ] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec0 as *const _)
            );
            let res1 = inp_vec1._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec1 as *const _)
            );

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
        }
    }
}

fn micro_kernel_3<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            ); // prettier-ignore

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec0 as *const _)
            );
            let res1 = inp_vec1._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec1 as *const _)
            );
            let res2 = inp_vec2._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec2 as *const _)
            );

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
        }
    }
}

fn micro_kernel_5<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            ); // prettier-ignore

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec4 = &mut out[co_offset + ofs + 4 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec0 as *const _)
            );
            let res1 = inp_vec1._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec1 as *const _)
            );
            let res2 = inp_vec2._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec2 as *const _)
            );
            let res3 = inp_vec3._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec3 as *const _)
            );
            let res4 = inp_vec4._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec4 as *const _)
            );

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
            out_vec3.write_unaligned(res3);
            out_vec4.write_unaligned(res4);
        }
    }
}

fn micro_kernel_1_with_buffer<T, const REGNUM: usize, const BUFFER_SIZE: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &[<T as TypeCommon>::Vec]
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T>
{
    let inp: &Pointer<T> = &inp;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors[0] as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);

            *out_vec0 = res0;
        }
    }
}

fn micro_kernel_2_with_buffer<T, const REGNUM: usize, const BUFFER_SIZE: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &[<T as TypeCommon>::Vec]
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T>
{
    let inp: &Pointer<T> = &inp;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors[0] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors[1] as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);
            let res1 = inp_vec1._mul_add(kernel_vec, *out_vec1);

            *out_vec0 = res0;
            *out_vec1 = res1;
        }
    }
}

fn micro_kernel_3_with_buffer<T, const REGNUM: usize, const BUFFER_SIZE: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &[<T as TypeCommon>::Vec]
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T>
{
    let inp: &Pointer<T> = &inp;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors.get_unchecked_mut(
                0
            ) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors.get_unchecked_mut(
                1
            ) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut res_vectors.get_unchecked_mut(
                2
            ) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);
            let res1 = inp_vec1._mul_add(kernel_vec, *out_vec1);
            let res2 = inp_vec2._mul_add(kernel_vec, *out_vec2);

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
        }
    }
}

fn micro_kernel_5_with_buffer<T, const REGNUM: usize, const BUFFER_SIZE: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &[<T as TypeCommon>::Vec]
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T>
{
    let inp: &Pointer<T> = &inp;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors.get_unchecked_mut(
                0
            ) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors.get_unchecked_mut(
                1
            ) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut res_vectors.get_unchecked_mut(
                2
            ) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = &mut res_vectors.get_unchecked_mut(
                3
            ) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = &mut res_vectors.get_unchecked_mut(
                4
            ) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);
            let res1 = inp_vec1._mul_add(kernel_vec, *out_vec1);
            let res2 = inp_vec2._mul_add(kernel_vec, *out_vec2);
            let res3 = inp_vec3._mul_add(kernel_vec, *out_vec3);
            let res4 = inp_vec4._mul_add(kernel_vec, *out_vec4);

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
            *out_vec4 = res4;
        }
    }
}

fn micro_kernel_6<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel
                    [

                            co_offset +
                            kernel_offset +
                            j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)

                    ] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec4 = &mut out[co_offset + ofs + 4 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec5 = &mut out[co_offset + ofs + 5 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec0 as *const _)
            );
            let res1 = inp_vec1._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec1 as *const _)
            );
            let res2 = inp_vec2._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec2 as *const _)
            );
            let res3 = inp_vec3._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec3 as *const _)
            );
            let res4 = inp_vec4._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec4 as *const _)
            );
            let res5 = inp_vec5._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec5 as *const _)
            );

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
            out_vec3.write_unaligned(res3);
            out_vec4.write_unaligned(res4);
            out_vec5.write_unaligned(res5);
        }
    }
}

fn micro_kernel_6_with_buffer<T, const REGNUM: usize, const BUFFER_SIZE: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &[<T as TypeCommon>::Vec]
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T>
{
    let inp: &Pointer<T> = &inp;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors[0] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors[1] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut res_vectors[2] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = &mut res_vectors[3] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = &mut res_vectors[4] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec5 = &mut res_vectors[5] as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);
            let res1 = inp_vec1._mul_add(kernel_vec, *out_vec1);
            let res2 = inp_vec2._mul_add(kernel_vec, *out_vec2);
            let res3 = inp_vec3._mul_add(kernel_vec, *out_vec3);
            let res4 = inp_vec4._mul_add(kernel_vec, *out_vec4);
            let res5 = inp_vec5._mul_add(kernel_vec, *out_vec5);

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
            *out_vec4 = res4;
            *out_vec5 = res5;
        }
    }
}

fn load_store_res_buffer<T, const REGNUM: usize, const LOAD: bool>(
    num_co_rb: i64,
    co_b_remain: i64,
    osw: i64,
    out_offset: i64,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    out: &mut Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    for j in 0..num_co_rb {
        let buffers = unsafe { res_buffer.get_unchecked_mut(j as usize) };
        for r in 0..REGNUM as i64 {
            unsafe {
                let out_ptr = &mut out[out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw] as *mut _ as *mut T; // prettier-ignore
                let buffer = buffers.get_unchecked_mut(r as usize) as *mut _ as *mut T; // prettier-ignore
                if LOAD {
                    std::ptr::copy_nonoverlapping(
                        out_ptr,
                        buffer,
                        <<T as TypeCommon>::Vec as VecSize>::SIZE
                    );
                } else {
                    std::ptr::copy_nonoverlapping(
                        buffer,
                        out_ptr,
                        <<T as TypeCommon>::Vec as VecSize>::SIZE
                    );
                }
            }
        }
    }
    let buffers = unsafe { res_buffer.get_unchecked_mut(num_co_rb as usize) };
    for r in 0..REGNUM as i64 {
        unsafe {
            let out_ptr = &mut out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw] as *mut _ as *mut T; // prettier-ignore
            let buffer = buffers.get_unchecked_mut(r as usize) as *mut _ as *mut T; // prettier-ignore
            if LOAD {
                std::ptr::copy_nonoverlapping(out_ptr, buffer, co_b_remain as usize);
            } else {
                std::ptr::copy_nonoverlapping(buffer, out_ptr, co_b_remain as usize);
            }
        }
    }
}

fn pack_kernel<T>(
    num_co_rb: i64,
    co_b_remain: i64,
    kernel_offset: i64,
    kernel: &Pointer<T>,
    kernel_buffer: &mut Vec<<T as TypeCommon>::Vec>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecSize
{
    for j in 0..num_co_rb {
        unsafe {
            std::ptr::copy_nonoverlapping(
                &kernel[kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _ as *const T, // prettier-ignore
                kernel_buffer.get_unchecked_mut(j as usize) as *mut _ as *mut T,
                <<T as TypeCommon>::Vec as VecSize>::SIZE
            );
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            &kernel[kernel_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _ as *const T, // prettier-ignore
            kernel_buffer.get_unchecked_mut(num_co_rb as usize) as *mut _ as *mut T,
            co_b_remain as usize
        );
    }
}

fn micro_kernel_4<T, const REGNUM: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    co_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel
                    [

                            co_offset +
                            kernel_offset +
                            j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)

                    ] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec0 as *const _)
            );
            let res1 = inp_vec1._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec1 as *const _)
            );
            let res2 = inp_vec2._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec2 as *const _)
            );
            let res3 = inp_vec3._mul_add(
                kernel_vec,
                <T as TypeCommon>::Vec::from_ptr(out_vec3 as *const _)
            );

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
            out_vec3.write_unaligned(res3);
        }
    }
}

fn micro_kernel_4_with_buffer<T, const REGNUM: usize, const BUFFER_SIZE: usize>(
    num_co_rb: i64,
    kp: i64,
    i: i64,
    inp_offset: i64,
    step_width: i64,
    isw: i64,
    inp: &Pointer<T>,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    kernel: &[<T as TypeCommon>::Vec]
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T>
{
    let inp: &Pointer<T> = &inp;
    let _k = kp * (REGNUM as i64) + 0;
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]); // prettier-ignore
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize); // prettier-ignore

            let out_vec0 = res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec1 = res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec2 = res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore
            let out_vec3 = res_vectors.get_unchecked_mut(3) as *mut _ as *mut <T as TypeCommon>::Vec; // prettier-ignore

            let res0 = inp_vec0._mul_add(kernel_vec, *out_vec0);
            let res1 = inp_vec1._mul_add(kernel_vec, *out_vec1);
            let res2 = inp_vec2._mul_add(kernel_vec, *out_vec2);
            let res3 = inp_vec3._mul_add(kernel_vec, *out_vec3);

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
        }
    }
}

#[allow(unused)]
fn do_calculate_v2<T, VEC, const REGNUM: usize, const VECSIZE: usize>(
    num_co_rb: i64,
    k: i64,
    i: i64,
    k_offset: i64,
    out_offset: i64,
    kernel_offset: i64,
    step_width: i64,
    isw: i64,
    osw: i64,
    inp: &Pointer<T>,
    out: &mut Pointer<T>,
    kernel: &Pointer<T>
)
    where T: CommonBounds, VEC: VecTrait<T> + Copy + Init<T>
{
    let kernel_vec0 = unsafe { *(&kernel[kernel_offset + 0 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec1 = unsafe { *(&kernel[kernel_offset + 1 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec2 = unsafe { *(&kernel[kernel_offset + 2 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec3 = unsafe { *(&kernel[kernel_offset + 3 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec4 = unsafe { *(&kernel[kernel_offset + 4 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec5 = unsafe { *(&kernel[kernel_offset + 5 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec6 = unsafe { *(&kernel[kernel_offset + 6 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec7 = unsafe { *(&kernel[kernel_offset + 7 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec8 = unsafe { *(&kernel[kernel_offset + 8 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore
    let kernel_vec9 = unsafe { *(&kernel[kernel_offset + 9 * (VECSIZE as i64)] as *const _ as *const VEC) }; // prettier-ignore

    for d in 0..REGNUM as i64 {
        let ofs = out_offset + k * (REGNUM as i64) * osw + d * osw;
        let inp_vec = VEC::splat(inp[(k_offset + k * (REGNUM as i64) + d) * step_width * isw + i]); // prettier-ignore
        let out_vec0 = &mut out[ofs + 0 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec1 = &mut out[ofs + 1 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec2 = &mut out[ofs + 2 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec3 = &mut out[ofs + 3 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec4 = &mut out[ofs + 4 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec5 = &mut out[ofs + 5 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec6 = &mut out[ofs + 6 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec7 = &mut out[ofs + 7 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec8 = &mut out[ofs + 8 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore
        let out_vec9 = &mut out[ofs + 9 * (VECSIZE as i64)] as *mut _ as *mut VEC; // prettier-ignore

        unsafe {
            let res0 = inp_vec._mul_add(kernel_vec0, out_vec0.read());
            let res1 = inp_vec._mul_add(kernel_vec1, out_vec1.read());
            let res2 = inp_vec._mul_add(kernel_vec2, out_vec2.read());
            let res3 = inp_vec._mul_add(kernel_vec3, out_vec3.read());
            let res4 = inp_vec._mul_add(kernel_vec4, out_vec4.read());
            let res5 = inp_vec._mul_add(kernel_vec5, out_vec5.read());
            let res6 = inp_vec._mul_add(kernel_vec6, out_vec6.read());
            let res7 = inp_vec._mul_add(kernel_vec7, out_vec7.read());
            let res8 = inp_vec._mul_add(kernel_vec8, out_vec8.read());
            let res9 = inp_vec._mul_add(kernel_vec9, out_vec9.read());

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
            *out_vec4 = res4;
            *out_vec5 = res5;
            *out_vec6 = res6;
            *out_vec7 = res7;
            *out_vec8 = res8;
            *out_vec9 = res9;
        }
    }
}
