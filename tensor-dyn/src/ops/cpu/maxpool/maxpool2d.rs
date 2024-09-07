use tensor_common::pointer::Pointer;
use tensor_common::shape::Shape;
use tensor_types::type_promote::NormalOut;
use tensor_types::vectors::traits::*;
use crate::ops::cpu::convolutions::conv_config::Conv2dConfig;
use crate::ops::cpu::convolutions::conv_config::KernelParamAlgo;
use crate::ops::cpu::kernels::maxpool_kernel::*;
use crate::tensor_base::_Tensor;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use rayon::prelude::*;
use tensor_types::dtype::TypeCommon;

impl<T> _Tensor<T>
    where
        T: CommonBounds + IntoScalar<T> + NormalOut<Output = T>,
        <T as TypeCommon>::Vec: VecTrait<T> +
            Copy +
            Init<T> +
            Send +
            Sync +
            VecSize +
            NormalOut<Output = <T as TypeCommon>::Vec>
{
    pub fn max_pool2d<S: Into<Shape>>(
        &self,
        kernel_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        config: Option<&Conv2dConfig<T>>
    ) -> anyhow::Result<_Tensor<T>> {
        use crate::CONV_REGNUM;
        let kernel_shape: Shape = kernel_shape.into();
        let img_shape = self.shape();
        let batch = img_shape[0];
        let img_height = img_shape[1];
        let img_width = img_shape[2];
        let out_channels = img_shape[3];
        let kernel_height = kernel_shape[0];
        let kernel_width = kernel_shape[1];
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
        let output = _Tensor::<T>::full(T::NEG_INF, [batch, out_height, out_width, out_channels])?;
        let out = output.ptr();
        let inp = img.ptr();

        let osb = output.strides()[0]; // batch
        let osh = output.strides()[1]; // height
        let osw = output.strides()[2]; // width

        let isb = img.strides()[0]; // batch
        let ish = img.strides()[1]; // height
        let isw = img.strides()[2]; // width

        let (_, co_b) = match config {
            Some(config) => (config.ci_block_size, config.co_block_size),
            None => {
                let config = {
                    Conv2dConfig::<T>::new(
                        out_channels,
                        1,
                        [kernel_height, kernel_width],
                        KernelParamAlgo::Greedy
                    )
                };
                (config.ci_block_size, config.co_block_size)
            }
        };

        assert!(co_b <= out_channels);
        let num_co_b = out_channels / co_b;
        let num_wo_b = out_width / (CONV_REGNUM as i64);

        let co_b_remain = out_channels % co_b;
        let wo_b_remain = out_width % (CONV_REGNUM as i64);
        let num_co_rb = co_b / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        assert!(co_b % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) == 0 || co_b == 1);

        let outer = batch * num_co_b * out_height;

        let inp_cpy = inp.clone();
        let case0 = move |
            b: i64,
            l: i64,
            c: i64,
            inner_size: i64,
            kernel_fn: fn(i64, i64, i64, i64, i64, i64, i64, i64, &Pointer<T>, &mut Pointer<T>),
            mut out: Pointer<T>
        | {
            for kp in 0..num_wo_b {
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        kernel_fn(
                            inner_size,
                            kp,
                            c * co_b + b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                            c * co_b,
                            b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                            step_width,
                            isw,
                            osw,
                            &inp_cpy,
                            &mut out
                        );
                    }
                }
            }
        };

        let inp_cpy = inp.clone();
        let case1_helper = move |
            b: i64,
            l: i64,
            c: i64,
            remain: i64,
            micro_kernel: fn(i64, i64, i64, i64, i64, i64, i64, i64, &Pointer<T>, &mut Pointer<T>),
            mut out: Pointer<T>
        | {
            for n in 0..kernel_height {
                for m in 0..kernel_width {
                    micro_kernel(
                        remain,
                        num_wo_b,
                        c * co_b + b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                        c * co_b,
                        b * osb + l * osh + num_wo_b * CONV_REGNUM as i64 * osw, // prettier-ignore
                        step_width,
                        isw,
                        osw,
                        &inp_cpy,
                        &mut out
                    );
                }
            }
        };

        let case1_helper = &case1_helper;
        // case1 is used to handle the remain part of the width block
        let case1 = move |b: i64, l: i64, c: i64, out: Pointer<T>| {
            match wo_b_remain {
                1 => {
                    case1_helper(b, l, c, num_co_rb, micro_kernel_1::<T>, out);
                }
                2 => {
                    case1_helper(b, l, c, num_co_rb, micro_kernel_2::<T>, out);
                }
                3 => {
                    case1_helper(b, l, c, num_co_rb, micro_kernel_3::<T>, out);
                }
                4 => {
                    case1_helper(b, l, c, num_co_rb, micro_kernel_4::<T>, out);
                }
                5 => {
                    case1_helper(b, l, c, num_co_rb, micro_kernel_5::<T>, out);
                }
                6 => {
                    case1_helper(b, l, c, num_co_rb, micro_kernel_6::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                7 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_7::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                8 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_8::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                9 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_9::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                10 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_10::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                11 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_11::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                12 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_12::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                13 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_13::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                14 => {
                    case1_helper(b, l, c, ip, num_co_rb, ci_b_remain, micro_kernel_14::<T>, out);
                }
                _ => unimplemented!(),
            }
        };

        // case1 is used to handle the remain part of the width block, but at the same time, it also handles the remain part of the output channel
        let case1_remain1 = move |b: i64, l: i64, c: i64, out: Pointer<T>| {
            match wo_b_remain {
                1 => {
                    case1_helper(b, l, c, 1, micro_kernel_1_1::<T>, out);
                }
                2 => {
                    case1_helper(b, l, c, 1, micro_kernel_2_1::<T>, out);
                }
                3 => {
                    case1_helper(b, l, c, 1, micro_kernel_3_1::<T>, out);
                }
                4 => {
                    case1_helper(b, l, c, 1, micro_kernel_4_1::<T>, out);
                }
                5 => {
                    case1_helper(b, l, c, 1, micro_kernel_5_1::<T>, out);
                }
                6 => {
                    case1_helper(b, l, c, 1, micro_kernel_6_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                7 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_7_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                8 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_8_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                9 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_9_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                10 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_10_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                11 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_11_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                12 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_12_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                13 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_13_1::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                14 => {
                    case1_helper(b, l, c, ip, ci_b_remain, 1, micro_kernel_14_1::<T>, out);
                }
                _ => unimplemented!(),
            }
        };

        let inp_cpy = inp.clone();

        let case0 = &case0;

        // case2 is used to handle remain part of out channels
        let case2 = move |b: i64, l: i64, c: i64, mut out: Pointer<T>| {
            let num_vec_size = co_b_remain / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            let remain = co_b_remain % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            if remain == 0 {
                for kp in 0..num_wo_b {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            micro_kernel_regnum::<T>(
                                num_vec_size,
                                kp,
                                num_co_b * co_b + b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                                num_co_b * co_b,
                                b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                                step_width,
                                isw,
                                osw,
                                &inp_cpy,
                                &mut out
                            );
                        }
                    }
                }
            } else {
                let mut res_buffer =
                    vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); CONV_REGNUM]; num_vec_size as usize + 1];
                for kp in 0..num_wo_b {
                    load_store_res_buffer::<T, CONV_REGNUM, true>(
                        num_vec_size,
                        remain,
                        osw,
                        c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                        &mut res_buffer,
                        &mut out
                    );
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            micro_kernel_regnum_with_buffer::<T>(
                                num_vec_size,
                                kp,
                                c * co_b + b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                                step_width,
                                isw,
                                &inp_cpy,
                                &mut res_buffer
                            );
                        }
                    }
                    load_store_res_buffer::<T, CONV_REGNUM, false>(
                        num_vec_size,
                        remain,
                        osw,
                        c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                        &mut res_buffer,
                        &mut out
                    );
                }
            }
        };

        let inp_cpy = inp.clone();
        let case1 = &case1;
        let case3_helper = move |
            b: i64,
            l: i64,
            c: i64,
            wo_b_remain: i64,
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
                &Pointer<T>,
                &mut Vec<Vec<<T as TypeCommon>::Vec>>
            ),
            fast_micro_kernel: fn(
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                &Pointer<T>,
                &mut Pointer<T>
            ),
            mut out: Pointer<T>
        | {
            let num_vec_size = co_b_remain / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            let remain = co_b_remain % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            if remain == 0 {
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        fast_micro_kernel(
                            num_vec_size,
                            num_wo_b,
                            c * co_b + b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                            c * co_b,
                            b * osb + l * osh + num_wo_b * CONV_REGNUM as i64 * osw, // prettier-ignore
                            step_width,
                            isw,
                            osw,
                            &inp_cpy,
                            &mut out
                        );
                    }
                }
            } else {
                let mut remain_buffer =
                    vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); wo_b_remain as usize]; num_vec_size as usize + 1];
                load_fn(
                    num_vec_size,
                    remain,
                    osw,
                    c * co_b + b * osb + l * osh + num_wo_b * (CONV_REGNUM as i64) * osw,
                    &mut remain_buffer,
                    &mut out
                );
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        micro_kernel(
                            num_vec_size,
                            num_wo_b,
                            c * co_b + b * isb + (l * step_height + n * dh) * ish + m * dw * isw,
                            step_width,
                            isw,
                            &inp,
                            &mut remain_buffer
                        );
                    }
                }
                store_fn(
                    num_vec_size,
                    remain,
                    osw,
                    c * co_b + b * osb + l * osh + num_wo_b * (CONV_REGNUM as i64) * osw,
                    &mut remain_buffer,
                    &mut out
                );
            }
        };

        let case3 = move |b: i64, l: i64, c: i64, out: Pointer<T>| {
            match wo_b_remain {
                1 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        1,
                        load_store_res_buffer::<T, 1, true>,
                        load_store_res_buffer::<T, 1, false>,
                        micro_kernel_1_with_buffer::<T>,
                        micro_kernel_1::<T>,
                        out
                    );
                }
                2 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        2,
                        load_store_res_buffer::<T, 2, true>,
                        load_store_res_buffer::<T, 2, false>,
                        micro_kernel_2_with_buffer::<T>,
                        micro_kernel_2::<T>,
                        out
                    );
                }
                3 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        3,
                        load_store_res_buffer::<T, 3, true>,
                        load_store_res_buffer::<T, 3, false>,
                        micro_kernel_3_with_buffer::<T>,
                        micro_kernel_3::<T>,
                        out
                    );
                }
                4 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        4,
                        load_store_res_buffer::<T, 4, true>,
                        load_store_res_buffer::<T, 4, false>,
                        micro_kernel_4_with_buffer::<T>,
                        micro_kernel_4::<T>,
                        out
                    );
                }
                5 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        5,
                        load_store_res_buffer::<T, 5, true>,
                        load_store_res_buffer::<T, 5, false>,
                        micro_kernel_5_with_buffer::<T>,
                        micro_kernel_5::<T>,
                        out
                    );
                }
                6 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        6,
                        load_store_res_buffer::<T, 6, true>,
                        load_store_res_buffer::<T, 6, false>,
                        micro_kernel_6_with_buffer::<T>,
                        micro_kernel_6::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                7 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        7,
                        load_store_res_buffer::<T, 7, true>,
                        load_store_res_buffer::<T, 7, false>,
                        micro_kernel_7_with_buffer::<T>,
                        micro_kernel_7::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                8 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        8,
                        load_store_res_buffer::<T, 8, true>,
                        load_store_res_buffer::<T, 8, false>,
                        micro_kernel_8_with_buffer::<T>,
                        micro_kernel_8::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                9 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        9,
                        load_store_res_buffer::<T, 9, true>,
                        load_store_res_buffer::<T, 9, false>,
                        micro_kernel_9_with_buffer::<T>,
                        micro_kernel_9::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                10 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        10,
                        load_store_res_buffer::<T, 10, true>,
                        load_store_res_buffer::<T, 10, false>,
                        micro_kernel_10_with_buffer::<T>,
                        micro_kernel_10::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                11 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        11,
                        load_store_res_buffer::<T, 11, true>,
                        load_store_res_buffer::<T, 11, false>,
                        micro_kernel_11_with_buffer::<T>,
                        micro_kernel_11::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                12 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        12,
                        load_store_res_buffer::<T, 12, true>,
                        load_store_res_buffer::<T, 12, false>,
                        micro_kernel_12_with_buffer::<T>,
                        micro_kernel_12::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                13 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        13,
                        load_store_res_buffer::<T, 13, true>,
                        load_store_res_buffer::<T, 13, false>,
                        micro_kernel_13_with_buffer::<T>,
                        micro_kernel_13::<T>,
                        out
                    );
                }
                #[cfg(target_feature = "avx512f")]
                14 => {
                    case3_helper(
                        b,
                        l,
                        c,
                        ip,
                        ci_b_remain,
                        14,
                        load_store_res_buffer::<T, 14, true>,
                        load_store_res_buffer::<T, 14, false>,
                        micro_kernel_14_with_buffer::<T>,
                        micro_kernel_14::<T>,
                        out
                    );
                }
                _ => unimplemented!(),
            }
        };
        (0..outer).into_iter().for_each(|idx| {
            let b = idx / (num_co_b * out_height);
            let c = (idx / out_height) % num_co_b;
            let l = idx % out_height;
            match (co_b_remain == 0, wo_b_remain == 0) {
                (true, true) => {
                    // co_b must be the multiple of vec_size, or else it will must be 1
                    if co_b > 1 {
                        case0(b, l, c, num_co_rb, micro_kernel_regnum::<T>, out.clone());
                    } else {
                        assert_eq!(co_b, 1);
                        case0(b, l, c, 1, micro_kernel_regnum_1::<T>, out.clone());
                    }
                }
                (true, false) => {
                    if co_b > 1 {
                        case0(b, l, c, num_co_rb, micro_kernel_regnum::<T>, out.clone());
                        case1(b, l, c, out.clone());
                    } else {
                        assert_eq!(co_b, 1);
                        case0(b, l, c, 1, micro_kernel_regnum_1::<T>, out.clone());
                        case1_remain1(b, l, c, out.clone());
                    }
                }
                (false, true) => {
                    assert!(co_b > 1);
                    match c < num_co_b - 1 {
                        true => {
                            case0(b, l, c, num_co_rb, micro_kernel_regnum::<T>, out.clone());
                        }
                        false => {
                            case0(b, l, c, num_co_rb, micro_kernel_regnum::<T>, out.clone());
                            case2(b, l, num_co_b, out.clone());
                        }
                    }
                }
                (false, false) => {
                    assert!(co_b > 1);
                    match c < num_co_b - 1 {
                        true => {
                            case0(b, l, c, num_co_rb, micro_kernel_regnum::<T>, out.clone());
                            case1(b, l, c, out.clone());
                        }
                        false => {
                            case0(b, l, c, num_co_rb, micro_kernel_regnum::<T>, out.clone());
                            case1(b, l, c, out.clone());
                            case2(b, l, num_co_b, out.clone());
                            case3(b, l, num_co_b, out.clone());
                        }
                    }
                }
            }
        });

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
