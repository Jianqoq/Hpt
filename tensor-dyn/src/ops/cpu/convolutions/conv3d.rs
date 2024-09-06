use tensor_common::err_handler::ErrHandler;
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
    pub fn conv3d(
        &self,
        kernels: &_Tensor<T>,
        steps: [i64; 3],
        padding: [(i64, i64); 3],
        dilation: [i64; 3]
    ) -> anyhow::Result<_Tensor<T>> {
        use tensor_common::shape_utils::mt_intervals;

        use crate::CONV_REGNUM;
        ErrHandler::check_ndim_match(self.ndim(), 5)?;
        let inp_shape = self.shape();
        let batch = inp_shape[0];
        let inp_depth = inp_shape[1];
        let inp_height = inp_shape[2];
        let inp_width = inp_shape[3];
        let inp_channels = inp_shape[4];
        let kernel_shape = kernels.shape();
        let kernel_depth = kernel_shape[0];
        let kernel_height = kernel_shape[1];
        let kernel_width = kernel_shape[2];
        let in_channels = kernel_shape[3];
        let out_channels = kernel_shape[4];
        if in_channels != inp_channels {
            panic!(
                "The number of input channels in the image must be equal to the number of input channels in the kernel."
            );
        }
        let [step_depth, step_width, step_height] = steps;
        let [
            (pd_start, pd_end),
            (ph_start, ph_end),
            (pw_start, pw_end),
        ] = padding;
        let [dd, dh, dw] = dilation;

        let out_height =
            (inp_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
        let out_width =
            (inp_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
        let out_depth =
            (inp_depth + pd_start + pd_end - dd * (kernel_depth - 1) - 1) / step_depth + 1;
        let inp = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
            self.pad(
                &[
                    (0, 0),
                    (pd_start, pd_end),
                    (ph_start, ph_end),
                    (pw_start, pw_end),
                    (0, 0),
                ],
                T::ZERO
            )?
        } else {
            self.clone()
        };
        let output = _Tensor::<T>::zeros([batch, out_depth, out_height, out_width, out_channels])?;
        let out = output.ptr();
        let inp_ptr = inp.ptr();
        let kernel = kernels.ptr();

        let osb = output.strides()[0]; // batch
        let osd = output.strides()[1]; // depth
        let osh = output.strides()[2]; // height
        let osw = output.strides()[3]; // width

        let isb = inp.strides()[0]; // batch
        let isd = inp.strides()[1]; // depth
        let ish = inp.strides()[2]; // height
        let isw = inp.strides()[3]; // width

        let ksd = kernels.strides()[0]; // kernel_depth
        let ks0 = kernels.strides()[1]; // kernel_height
        let ks1 = kernels.strides()[2]; // kernel_width
        let ks2 = kernels.strides()[3]; // in_channels

        let l1_cache =
            cache_size::l1_cache_size().unwrap_or(32 * 1024 /* 32 kb */) / std::mem::size_of::<T>();

        let (co_b, ci_b) = find_exact_combination::<T, CONV_REGNUM>(
            l1_cache as i64,
            out_channels as i64,
            in_channels as i64,
            kernel_height as i64,
            kernel_width as i64,
            kernel_depth as i64
        );
        let num_co_b = out_channels / co_b;
        let num_wo_b = out_width / (CONV_REGNUM as i64);
        let num_ci_b = in_channels / ci_b;

        let co_b_remain = out_channels % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        let wo_b_remain = out_width % (CONV_REGNUM as i64);
        let ci_b_remain = in_channels % ci_b;
        let num_co_rb = co_b / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);

        let outer = batch * num_co_b * num_ci_b * out_depth * out_height;

        let inp_cpy = inp_ptr.clone();
        let kernel_cpy = kernel.clone();
        let case0 = move |
            b: i64,
            l: i64,
            d: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            mut out: Pointer<T>
        | {
            for kp in 0..num_wo_b {
                for p in 0..kernel_depth {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            for ii in 0..ci_b_remain {
                                let i = ip * ci_b + ii;
                                micro_kernel_regnum::<T, CONV_REGNUM>(
                                    num_co_rb,
                                    kp,
                                    i,
                                    b * isb +
                                        (l * step_height + n * dh) * ish +
                                        (d * step_depth + p * dd) * isd +
                                        m * dw * isw,
                                    c * co_b,
                                    b * osb + l * osh + kp * CONV_REGNUM as i64 * osw, // prettier-ignore
                                    n * ks0 + m * ks1 + i * ks2 + p * ksd,
                                    step_width,
                                    isw,
                                    osw,
                                    &inp_cpy,
                                    &mut out,
                                    &kernel_cpy
                                );
                            }
                        }
                    }
                }
            }
        };

        let inp_cpy = inp_ptr.clone();
        let kernel_cpy = kernel.clone();
        let case1_helper = move |
            b: i64,
            l: i64,
            d: i64,
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
            for p in 0..kernel_depth {
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for ii in 0..ci_b_remain {
                            let i = ip * ci_b + ii;
                            micro_kernel(
                                num_co_rb,
                                num_wo_b,
                                i,
                                b * isb +
                                    (l * step_height + n * dh) * ish +
                                    (d * step_depth + p * dd) * isd +
                                    m * isw,
                                c * co_b,
                                b * osb + l * osh + d * osd + num_wo_b * CONV_REGNUM as i64 * osw, // prettier-ignore
                                n * ks0 + m * ks1 + i * ks2 + p * ksd,
                                step_width,
                                isw,
                                osw,
                                &inp_cpy,
                                &mut out,
                                &kernel_cpy
                            );
                        }
                    }
                }
            }
        };

        let case1 = move |
            b: i64,
            l: i64,
            d: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            out: Pointer<T>
        | {
            match wo_b_remain {
                2 => {
                    case1_helper(
                        b,
                        l,
                        d,
                        c,
                        ip,
                        ci_b_remain,
                        micro_kernel_2::<T, CONV_REGNUM>,
                        out
                    );
                }
                4 => {
                    case1_helper(
                        b,
                        l,
                        d,
                        c,
                        ip,
                        ci_b_remain,
                        micro_kernel_4::<T, CONV_REGNUM>,
                        out
                    );
                }
                6 => {
                    case1_helper(
                        b,
                        l,
                        d,
                        c,
                        ip,
                        ci_b_remain,
                        micro_kernel_6::<T, CONV_REGNUM>,
                        out
                    );
                }
                _ => unimplemented!(),
            }
        };

        let inp_cpy = inp_ptr.clone();
        let kernel_cpy = kernel.clone();
        let case2 = move |
            b: i64,
            l: i64,
            d: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            mut out: Pointer<T>
        | {
            let mut res_buffer =
                vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); CONV_REGNUM]; num_co_rb as usize + 1];
            for kp in 0..num_wo_b {
                load_store_res_buffer::<T, CONV_REGNUM, true>(
                    num_co_rb,
                    co_b_remain,
                    osw,
                    c * co_b + b * osb + l * osh + d * osd + kp * (CONV_REGNUM as i64) * osw,
                    &mut res_buffer,
                    &mut out
                );
                for p in 0..kernel_depth {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            for ii in 0..ci_b_remain {
                                let i = ip * ci_b + ii;
                                // need either packing or minimize the cost of cache miss
                                micro_kernel_regnum_with_buffer::<T, CONV_REGNUM>(
                                    num_co_rb,
                                    kp,
                                    i,
                                    b * isb +
                                        (l * step_height + n * dh) * ish +
                                        (d * step_depth + p * dd) * isd +
                                        m * dw * isw,
                                    c * co_b,
                                    n * ks0 + m * ks1 + i * ks2 + p * ksd,
                                    step_width,
                                    isw,
                                    &inp_cpy,
                                    &mut res_buffer,
                                    &kernel_cpy
                                );
                            }
                        }
                    }
                }
                load_store_res_buffer::<T, CONV_REGNUM, false>(
                    num_co_rb,
                    co_b_remain,
                    osw,
                    c * co_b + b * osb + l * osh + d * osd + kp * (CONV_REGNUM as i64) * osw,
                    &mut res_buffer,
                    &mut out
                );
            }
        };

        let case3_helper = move |
            b: i64,
            l: i64,
            d: i64,
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
            load_fn(
                num_co_rb,
                co_b_remain,
                osw,
                c * co_b + b * osb + l * osh + d * osd + num_wo_b * (CONV_REGNUM as i64) * osw,
                &mut remain_buffer,
                &mut out
            );
            for p in 0..kernel_depth {
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for ii in 0..ci_b_remain {
                            let i = ip * ci_b + ii;
                            pack_kernel(
                                num_co_rb,
                                co_b_remain,
                                c * co_b + n * ks0 + m * ks1 + i * ks2 + p * ksd,
                                &kernel,
                                &mut kernel_buffer
                            );
                            micro_kernel(
                                num_co_rb,
                                num_wo_b,
                                i,
                                b * isb +
                                    (l * step_height + n * dh) * ish +
                                    (d * step_depth + p * dd) * ish +
                                    m * isw,
                                step_width,
                                isw,
                                &inp_ptr,
                                &mut remain_buffer,
                                &kernel_buffer
                            );
                        }
                    }
                }
            }
            store_fn(
                num_co_rb,
                co_b_remain,
                osw,
                c * co_b + b * osb + l * osh + d * osd + num_wo_b * (CONV_REGNUM as i64) * osw,
                &mut remain_buffer,
                &mut out
            );
        };

        let case3 = move |
            b: i64,
            l: i64,
            d: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            out: Pointer<T>
        | {
            match wo_b_remain {
                1 => {
                    case3_helper(
                        b,
                        l,
                        d,
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
                        d,
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
                        d,
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
                        d,
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
                        d,
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
                        d,
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
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    case0(b, l, d, c, ip, ci_b, out.clone());
                });
            }
            (true, false, true) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    case0(b, l, d, c, ip, ci_b, out.clone());
                    case1(b, l, d, c, ip, ci_b, out.clone());
                });
            }
            (false, true, true) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    if c < num_co_b - 1 {
                        case0(b, l, d, c, ip, ci_b, out.clone());
                    } else {
                        case2(b, l, d, c, ip, ci_b, out.clone());
                    }
                });
            }
            (false, false, true) => {
                let intervals = mt_intervals(outer as usize, outer as usize);
                intervals.into_par_iter().for_each(|(start, end)| {
                    for idx in start as i64..end as i64 {
                        let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                        let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                        let ip = (idx / (out_depth * out_height)) % num_ci_b;
                        let d = (idx / out_height) % out_depth;
                        let l = idx % out_height;
                        if c < num_co_b - 1 {
                            case0(b, l, d, c, ip, ci_b, out.clone());
                            case1(b, l, d, c, ip, ci_b, out.clone());
                        } else {
                            case2(b, l, d, c, ip, ci_b, out.clone());
                            case3(b, l, d, c, ip, ci_b, out.clone());
                        }
                    }
                });
            }
            (true, true, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        case0(b, l, d, c, ip, ci_b, out.clone());
                    } else {
                        // when ip == num_ci_b - 1, we first need to process not remain part, then remain part
                        case0(b, l, d, c, ip, ci_b, out.clone());
                        case0(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                    }
                });
            }
            (true, false, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        case0(b, l, d, c, ip, ci_b, out.clone());
                        case1(b, l, d, c, ip, ci_b, out.clone());
                    } else {
                        // when ip == num_ci_b - 1, we first need to process not remain part, then remain part
                        case0(b, l, d, c, ip, ci_b, out.clone());
                        case1(b, l, d, c, ip, ci_b, out.clone());
                        case0(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                        case1(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                    }
                });
            }
            (false, true, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        if c < num_co_b - 1 {
                            case0(b, l, d, c, ip, ci_b, out.clone());
                        } else {
                            case2(b, l, d, c, ip, ci_b, out.clone());
                        }
                    } else {
                        // when ip == num_ci_b - 1, we first need to process not remain part, then remain part
                        if c < num_co_b - 1 {
                            case0(b, l, d, c, ip, ci_b, out.clone());
                            case0(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                        } else {
                            case2(b, l, d, c, ip, ci_b, out.clone());
                            case2(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                        }
                    }
                });
            }
            (false, false, false) => {
                (0..outer).into_par_iter().for_each(|idx| {
                    let b = idx / (num_co_b * num_ci_b * out_depth * out_height);
                    let c = (idx / (num_ci_b * out_depth * out_height)) % num_co_b;
                    let ip = (idx / (out_depth * out_height)) % num_ci_b;
                    let d = (idx / out_height) % out_depth;
                    let l = idx % out_height;
                    if ip < num_ci_b - 1 {
                        if c < num_co_b - 1 {
                            case0(b, l, d, c, ip, ci_b, out.clone());
                            case1(b, l, d, c, ip, ci_b, out.clone());
                        } else {
                            case2(b, l, d, c, ip, ci_b, out.clone());
                            case3(b, l, d, c, ip, ci_b, out.clone());
                        }
                    } else {
                        if c < num_co_b - 1 {
                            case0(b, l, d, c, ip, ci_b, out.clone());
                            case1(b, l, d, c, ip, ci_b, out.clone());
                            case0(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                            case1(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                        } else {
                            case2(b, l, d, c, ip, ci_b, out.clone());
                            case3(b, l, d, c, ip, ci_b, out.clone());
                            case2(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                            case3(b, l, d, c, in_channels / ci_b, ci_b_remain, out.clone());
                        }
                    }
                });
            }
        }
        Ok(output)
    }
}

fn find_exact_combination<T: CommonBounds, const REGNUM: usize>(
    max_cache_size: i64,
    max_co_b: i64,
    max_ci_b: i64,
    weight_size: i64,
    height_size: i64,
    depth_size: i64
) -> (i64, i64)
    where <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let mut best_co_b = 0;
    let mut best_ci_b = 0;

    for co_b in (1..max_co_b + 1)
        .rev()
        .filter(|&co_b| co_b % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) == 0) {
        // 只遍历 wo_b 是 7 的倍数的情况
        for ci_b in (1..max_ci_b + 1).rev() {
            let product =
                co_b * (REGNUM as i64) +
                weight_size * height_size * depth_size * ci_b * ((REGNUM as i64) + co_b);

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

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 6;
    let inp_vec6 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = &mut out[co_offset + ofs + 4 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec5 = &mut out[co_offset + ofs + 5 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec6 = &mut out[co_offset + ofs + 6 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;

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

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 6;
    let inp_vec6 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(&kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = res_vectors.get_unchecked_mut(3) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = res_vectors.get_unchecked_mut(4) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec5 = res_vectors.get_unchecked_mut(5) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec6 = res_vectors.get_unchecked_mut(6) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());
            let res3 = inp_vec3._mul_add(kernel_vec, out_vec3.read_unaligned());
            let res4 = inp_vec4._mul_add(kernel_vec, out_vec4.read_unaligned());
            let res5 = inp_vec5._mul_add(kernel_vec, out_vec5.read_unaligned());
            let res6 = inp_vec6._mul_add(kernel_vec, out_vec6.read_unaligned());

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

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());

            *out_vec0 = res0;
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());

            *out_vec0 = res0;
            *out_vec1 = res1;
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = &mut res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = &mut res_vectors.get_unchecked_mut(3) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = &mut res_vectors.get_unchecked_mut(4) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());
            let res3 = inp_vec3._mul_add(kernel_vec, out_vec3.read_unaligned());
            let res4 = inp_vec4._mul_add(kernel_vec, out_vec4.read_unaligned());

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
            *out_vec4 = res4;
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = &mut out[co_offset + ofs + 4 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec5 = &mut out[co_offset + ofs + 5 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());
            let res3 = inp_vec3._mul_add(kernel_vec, out_vec3.read_unaligned());
            let res4 = inp_vec4._mul_add(kernel_vec, out_vec4.read_unaligned());
            let res5 = inp_vec5._mul_add(kernel_vec, out_vec5.read_unaligned());

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
            out_vec3.write_unaligned(res3);
            out_vec4.write_unaligned(res4);
            out_vec5.write_unaligned(res5);
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 4;
    let inp_vec4 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 5;
    let inp_vec5 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = res_vectors.get_unchecked_mut(3) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 = res_vectors.get_unchecked_mut(4) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec5 = res_vectors.get_unchecked_mut(5) as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());
            let res3 = inp_vec3._mul_add(kernel_vec, out_vec3.read_unaligned());
            let res4 = inp_vec4._mul_add(kernel_vec, out_vec4.read_unaligned());
            let res5 = inp_vec5._mul_add(kernel_vec, out_vec5.read_unaligned());

            *out_vec0 = res0;
            *out_vec1 = res1;
            *out_vec2 = res2;
            *out_vec3 = res3;
            *out_vec4 = res4;
            *out_vec5 = res5;
        }
    }
}

#[rustfmt::skip]
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

#[rustfmt::skip]
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
                &kernel[kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _ as *const T,
                kernel_buffer.get_unchecked_mut(j as usize) as *mut _ as *mut T,
                <<T as TypeCommon>::Vec as VecSize>::SIZE
            );
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            &kernel[kernel_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _ as *const T,
            kernel_buffer.get_unchecked_mut(num_co_rb as usize) as *mut _ as *mut T,
            co_b_remain as usize
        );
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb {
        let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
            );

            let out_vec0 = &mut out[co_offset + ofs + 0 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = &mut out[co_offset + ofs + 1 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = &mut out[co_offset + ofs + 2 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = &mut out[co_offset + ofs + 3 * osw] as *mut _ as *mut <T as TypeCommon>::Vec;

            let res0 = inp_vec0._mul_add(kernel_vec, out_vec0.read_unaligned());
            let res1 = inp_vec1._mul_add(kernel_vec, out_vec1.read_unaligned());
            let res2 = inp_vec2._mul_add(kernel_vec, out_vec2.read_unaligned());
            let res3 = inp_vec3._mul_add(kernel_vec, out_vec3.read_unaligned());

            out_vec0.write_unaligned(res0);
            out_vec1.write_unaligned(res1);
            out_vec2.write_unaligned(res2);
            out_vec3.write_unaligned(res3);
        }
    }
}

#[rustfmt::skip]
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
    let inp_vec0 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 1;
    let inp_vec1 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 2;
    let inp_vec2 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    let _k = kp * (REGNUM as i64) + 3;
    let inp_vec3 = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
    for j in 0..num_co_rb + 1 {
        unsafe {
            let kernel_vec = *kernel.get_unchecked(j as usize);
            let res_vectors = res_buffer.get_unchecked_mut(j as usize);

            let out_vec0 = res_vectors.get_unchecked_mut(0) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 = res_vectors.get_unchecked_mut(1) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 = res_vectors.get_unchecked_mut(2) as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 = res_vectors.get_unchecked_mut(3) as *mut _ as *mut <T as TypeCommon>::Vec;

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
