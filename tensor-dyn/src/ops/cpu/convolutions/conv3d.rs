use tensor_common::err_handler::ErrHandler;
use tensor_common::pointer::Pointer;
use tensor_types::vectors::traits::*;
use crate::ops::cpu::convolutions::conv_config::KernelParamAlgo;
use crate::ops::cpu::kernels::conv2d_kernels::*;
use crate::tensor_base::_Tensor;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use rayon::prelude::*;
use num::traits::MulAdd;
use tensor_types::dtype::TypeCommon;

use super::conv_config::Conv3dConfig;

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
        dilation: [i64; 3],
        config: &Option<Conv3dConfig<T>>
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

        let (ci_b, co_b) = match config {
            Some(config) => (config.ci_block_size, config.co_block_size),
            None => {
                let config = {
                    Conv3dConfig::<T>::new(
                        out_channels,
                        in_channels,
                        [kernel_height, kernel_width, kernel_depth],
                        KernelParamAlgo::Greedy
                    )
                };
                (config.ci_block_size, config.co_block_size)
            }
        };
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
                                micro_kernel_regnum::<T>(
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
                    case1_helper(b, l, d, c, ip, ci_b_remain, micro_kernel_2::<T>, out);
                }
                4 => {
                    case1_helper(b, l, d, c, ip, ci_b_remain, micro_kernel_4::<T>, out);
                }
                6 => {
                    case1_helper(b, l, d, c, ip, ci_b_remain, micro_kernel_6::<T>, out);
                }
                _ => unimplemented!(),
            }
        };

        let inp_cpy = inp_ptr.clone();
        let kernel_cpy = kernel.clone();
        let case0 = &case0;
        let case2 = move |
            b: i64,
            l: i64,
            d: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            mut out: Pointer<T>
        | {
            let num_vec_size = co_b_remain / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            let remain = co_b_remain % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            if remain == 0 {
                case0(b, l, d, c, ip, ci_b, out.clone());
                for kp in 0..num_wo_b {
                    for p in 0..kernel_depth {
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                for ii in 0..ci_b_remain {
                                    let i = ip * ci_b + ii;
                                    micro_kernel_regnum::<T>(
                                        num_vec_size,
                                        kp,
                                        i,
                                        b * isb +
                                            (l * step_height + n * dh) * ish +
                                            (d * step_depth + p * dd) * isd +
                                            m * dw * isw,
                                        num_co_b * co_b,
                                        b * osb + l * osh + kp * CONV_REGNUM as i64 * osw, // prettier-ignore
                                        n * ks0 + m * ks1 + i * ks2,
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
            } else {
                let mut res_buffer =
                    vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); CONV_REGNUM]; num_co_rb as usize + 1];
                let mut kernel_buffer =
                    vec![<T as TypeCommon>::Vec::splat(T::ZERO); num_vec_size as usize + 1];
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
                                    pack_kernel(
                                        num_vec_size,
                                        remain,
                                        c * co_b + n * ks0 + m * ks1 + i * ks2 + p * ksd,
                                        &kernel_cpy,
                                        &mut kernel_buffer
                                    );
                                    micro_kernel_regnum_with_buffer::<T>(
                                        num_vec_size,
                                        kp,
                                        i,
                                        b * isb +
                                            (d * step_depth + p * dd) * isd +
                                            (l * step_height + n * dh) * ish +
                                            m * dw * isw,
                                        step_width,
                                        isw,
                                        &inp_cpy,
                                        &mut res_buffer,
                                        &kernel_buffer
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
                        micro_kernel_1_with_buffer::<T>,
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
                        micro_kernel_2_with_buffer::<T>,
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
                        micro_kernel_3_with_buffer::<T>,
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
                        micro_kernel_4_with_buffer::<T>,
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
                        micro_kernel_5_with_buffer::<T>,
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
                        micro_kernel_6_with_buffer::<T>,
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
