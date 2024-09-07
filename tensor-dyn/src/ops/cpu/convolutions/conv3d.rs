use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_common::pointer::Pointer;
use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_types::{
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    traits::{ Init, VecSize, VecTrait },
};
use tensor_common::err_handler::ErrHandler::InvalidCacheParam;
use tensor_common::err_handler::ErrHandler::InvalidInputShape;
use tensor_traits::TensorCreator;
use crate::{
    ops::cpu::{
        convolutions::conv_config::KernelParamAlgo,
        kernels::conv_kernels::micro_kernel_regnum,
    },
    tensor_base::_Tensor,
};
use crate::ops::cpu::kernels::conv_kernels::*;
use super::conv_config::Conv3dConfig;

impl<T> _Tensor<T>
    where
        T: CommonBounds + IntoScalar<T>,
        <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + Send + Sync + VecSize
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn conv3d(
        &self,
        kernels: &_Tensor<T>,
        steps: [i64; 3],
        padding: [(i64, i64); 3],
        dilation: [i64; 3],
        config: Option<&Conv3dConfig<T>>
    ) -> anyhow::Result<_Tensor<T>> {
        use crate::CONV_REGNUM;

        let img_shape = self.shape();
        let batch = img_shape[0];
        let img_depth = img_shape[1];
        let img_height = img_shape[2];
        let img_width = img_shape[3];
        let img_channels = img_shape[4];
        let kernel_shape = kernels.shape();
        let kernel_depth = kernel_shape[0];
        let kernel_height = kernel_shape[1];
        let kernel_width = kernel_shape[2];
        let in_channels = kernel_shape[3];
        let out_channels = kernel_shape[4];
        if in_channels != img_channels {
            panic!(
                "The number of input channels in the image must be equal to the number of input channels in the kernel."
            );
        }
        let [step_depth, step_height, step_width] = steps;
        let [
            (pd_start, pd_end),
            (ph_start, ph_end),
            (pw_start, pw_end),
        ] = padding;
        let [dd, dh, dw] = dilation;

        let out_height =
            (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
        let out_width =
            (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
        let out_depth =
            (img_depth + pd_start + pd_end - dd * (kernel_depth - 1) - 1) / step_depth + 1;
        let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
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
        if out_height <= 0 || out_width <= 0 || out_depth <= 0 {
            if out_height <= 0 {
                return Err(InvalidInputShape(out_height, core::panic::Location::caller()).into());
            } else if out_width <= 0 {
                return Err(InvalidInputShape(out_width, core::panic::Location::caller()).into());
            } else {
                return Err(InvalidInputShape(out_depth, core::panic::Location::caller()).into());
            }
        }
        let output = _Tensor::<T>::zeros([batch, out_depth, out_height, out_width, out_channels])?;
        let out = output.ptr();
        let inp = img.ptr();
        let kernel = kernels.ptr();

        let osb = output.strides()[0]; // batch
        let osd = output.strides()[1]; // depth
        let osh = output.strides()[2]; // height
        let osw = output.strides()[3]; // width

        let isb = img.strides()[0]; // batch
        let isd = img.strides()[1]; // depth
        let ish = img.strides()[2]; // height
        let isw = img.strides()[3]; // width

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

        let co_b_remain = out_channels % co_b;
        let wo_b_remain = out_width % (CONV_REGNUM as i64);
        let ci_b_remain = in_channels % ci_b;
        let num_co_rb = co_b / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        if !(co_b % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) == 0 || co_b == 1) || co_b > out_channels {
            return Err(
                InvalidCacheParam(
                    "co_b",
                    out_channels,
                    <<T as TypeCommon>::Vec as VecSize>::SIZE as i64,
                    co_b,
                    core::panic::Location::caller()
                ).into()
            );
        }

        let outer = batch * num_co_b * out_height * out_depth;

        let inp_cpy = inp.clone();
        let kernel_cpy = kernel.clone();
        let case0 = move |
            b: i64,
            l: i64,
            v: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            mut out: Pointer<T>
        | {
            for kp in 0..num_wo_b {
                for d in 0..kernel_depth {
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
                                        (v * step_depth + d * dd) * isd +
                                        m * dw * isw,
                                    c * co_b,
                                    b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw + d * osd,
                                    n * ks0 + m * ks1 + i * ks2 + d * ksd,
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

        let inp_cpy = inp.clone();
        let kernel_cpy = kernel.clone();
        let case1_helper = move |
            b: i64,
            l: i64,
            v: i64,
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
            for d in 0..kernel_depth {
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
                                    (v * step_depth + d * dd) * isd +
                                    m * dw * isw,
                                c * co_b,
                                b * osb + l * osh + num_wo_b * (CONV_REGNUM as i64) * osw + d * osd,
                                n * ks0 + m * ks1 + i * ks2 + d * ksd,
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
            v: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            out: Pointer<T>
        | {
            match wo_b_remain {
                1 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_1::<T>, out);
                }
                2 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_2::<T>, out);
                }
                3 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_3::<T>, out);
                }
                4 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_4::<T>, out);
                }
                5 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_5::<T>, out);
                }
                6 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_6::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                7 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_7::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                8 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_8::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                9 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_9::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                10 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_10::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                11 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_11::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                12 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_12::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                13 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_13::<T>, out);
                }
                #[cfg(target_feature = "avx512f")]
                14 => {
                    case1_helper(b, l, v, c, ip, ci_b_remain, micro_kernel_14::<T>, out);
                }
                _ => unimplemented!(),
            }
        };

        let inp_cpy = inp.clone();
        let kernel_cpy = kernel.clone();

        let case0 = &case0;
        let case2 = move |
            b: i64,
            l: i64,
            v: i64,
            c: i64,
            ip: i64,
            ci_b_remain: i64,
            mut out: Pointer<T>
        | {
            let num_vec_size = co_b_remain / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            let remain = co_b_remain % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            if remain == 0 {
                for kp in 0..num_wo_b {
                    for d in 0..kernel_depth {
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
                                            (v * step_depth + d * dd) * isd +
                                            m * dw * isw,
                                        num_co_b * co_b,
                                        b * osb +
                                            l * osh +
                                            kp * (CONV_REGNUM as i64) * osw +
                                            d * osd,
                                        n * ks0 + m * ks1 + i * ks2 + d * ksd,
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
                let mut kernel_buffer =
                    vec![<T as TypeCommon>::Vec::splat(T::ZERO); num_vec_size as usize + 1];
                let mut res_buffer =
                    vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); CONV_REGNUM]; num_vec_size as usize + 1];
                for kp in 0..num_wo_b {
                    load_store_res_buffer::<T, CONV_REGNUM, true>(
                        num_vec_size,
                        remain,
                        osw,
                        c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw + v * osd,
                        &mut res_buffer,
                        &mut out
                    );
                    for d in 0..kernel_depth {
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                for ii in 0..ci_b_remain {
                                    let i = ip * ci_b + ii;
                                    pack_kernel(
                                        num_vec_size,
                                        remain,
                                        c * co_b + n * ks0 + m * ks1 + i * ks2 + d * ksd,
                                        &kernel_cpy,
                                        &mut kernel_buffer
                                    );
                                    micro_kernel_regnum_with_buffer::<T>(
                                        num_vec_size,
                                        kp,
                                        i,
                                        b * isb +
                                            (l * step_height + n * dh) * ish +
                                            (v * step_depth + d * dd) * isd +
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
                        num_vec_size,
                        remain,
                        osw,
                        c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw + v * osd,
                        &mut res_buffer,
                        &mut out
                    );
                }
            }
        };

        let inp_cpy = inp.clone();
        let kernel_cpy = kernel.clone();
        let case1 = &case1;
        let case3_helper = move |
            b: i64,
            l: i64,
            v: i64,
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
            fast_micro_kernel: fn(
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
            let num_vec_size = co_b_remain / (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            let remain = co_b_remain % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
            if remain == 0 {
                for d in 0..kernel_depth {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            for ii in 0..ci_b_remain {
                                let i = ip * ci_b + ii;
                                fast_micro_kernel(
                                    num_vec_size,
                                    num_wo_b,
                                    i,
                                    b * isb +
                                        (l * step_height + n * dh) * ish +
                                        (v * step_depth + d * dd) * isd +
                                        m * dw * isw,
                                    c * co_b,
                                    b * osb +
                                        l * osh +
                                        num_wo_b * (CONV_REGNUM as i64) * osw +
                                        d * osd,
                                    n * ks0 + m * ks1 + i * ks2 + d * ksd,
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
            } else {
                let mut kernel_buffer =
                    vec![<T as TypeCommon>::Vec::splat(T::ZERO); num_vec_size as usize + 1];
                let mut remain_buffer =
                    vec![vec![<T as TypeCommon>::Vec::splat(T::ZERO); wo_b_remain as usize]; num_vec_size as usize + 1];
                load_fn(
                    num_vec_size,
                    remain,
                    osw,
                    c * co_b + b * osb + l * osh + num_wo_b * (CONV_REGNUM as i64) * osw + v * osd,
                    &mut remain_buffer,
                    &mut out
                );
                for d in 0..kernel_depth {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            for ii in 0..ci_b_remain {
                                let i = ip * ci_b + ii;
                                pack_kernel(
                                    num_vec_size,
                                    remain,
                                    c * co_b + n * ks0 + m * ks1 + i * ks2 + d * ksd,
                                    &kernel,
                                    &mut kernel_buffer
                                );
                                micro_kernel(
                                    num_vec_size,
                                    num_wo_b,
                                    i,
                                    b * isb +
                                        (l * step_height + n * dh) * ish +
                                        (v * step_depth + d * dd) * isd +
                                        m * dw * isw,
                                    step_width,
                                    isw,
                                    &inp,
                                    &mut remain_buffer,
                                    &kernel_buffer
                                );
                            }
                        }
                    }
                }
                store_fn(
                    num_vec_size,
                    remain,
                    osw,
                    c * co_b + b * osb + l * osh + num_wo_b * (CONV_REGNUM as i64) * osw + v * osd,
                    &mut remain_buffer,
                    &mut out
                );
            }
        };

        let case3 = move |
            b: i64,
            l: i64,
            v: i64,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        1,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        2,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        3,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        4,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        5,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        6,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        7,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        8,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        9,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        10,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        11,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        12,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        13,
                        pack_kernel::<T>,
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
                        v,
                        c,
                        ip,
                        ci_b_remain,
                        14,
                        pack_kernel::<T>,
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
        (0..outer).into_par_iter().for_each(|idx| {
            let b = idx / (num_co_b * out_height * out_depth);
            let c = ((idx / out_height) * out_depth) % num_co_b;
            let l = (idx / out_depth) % out_height;
            let v = idx % out_depth;
            match (co_b_remain == 0, wo_b_remain == 0, ci_b_remain == 0) {
                (true, true, true) => {
                    for ip in 0..num_ci_b {
                        case0(b, l, v, c, ip, ci_b, out.clone());
                    }
                }
                (true, true, false) => {
                    for ip in 0..num_ci_b {
                        case0(b, l, v, c, ip, ci_b, out.clone());
                    }
                    case0(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                }
                (true, false, true) => {
                    for ip in 0..num_ci_b {
                        case0(b, l, v, c, ip, ci_b, out.clone());
                        case1(b, l, v, c, ip, ci_b, out.clone());
                    }
                }
                (true, false, false) => {
                    for ip in 0..num_ci_b {
                        case0(b, l, v, c, ip, ci_b, out.clone());
                        case1(b, l, v, c, ip, ci_b, out.clone());
                    }
                    case0(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                    case1(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                }
                (false, true, true) => {
                    if c < num_co_b - 1 {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                        }
                    } else {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                            case2(b, l, v, num_co_b, ip, ci_b, out.clone());
                        }
                    }
                }
                (false, true, false) => {
                    if c < num_co_b - 1 {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                        }
                        case0(b, l, v, c, num_ci_b, ci_b, out.clone());
                    } else {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                            case2(b, l, v, num_co_b, ip, ci_b, out.clone());
                        }
                        case0(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                        case2(b, l, v, num_co_b, num_ci_b, ci_b_remain, out.clone());
                    }
                }
                (false, false, true) => {
                    if c < num_co_b - 1 {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                            case1(b, l, v, c, ip, ci_b, out.clone());
                        }
                    } else {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                            case1(b, l, v, c, ip, ci_b, out.clone());
                            case2(b, l, v, num_co_b, ip, ci_b, out.clone());
                            case3(b, l, v, num_co_b, ip, ci_b, out.clone());
                        }
                    }
                }
                (false, false, false) => {
                    if c < num_co_b - 1 {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                            case1(b, l, v, c, ip, ci_b, out.clone());
                        }
                        case0(b, l, v, c, num_ci_b, ci_b, out.clone());
                        case1(b, l, v, c, num_ci_b, ci_b, out.clone());
                        case0(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                        case1(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                    } else {
                        for ip in 0..num_ci_b {
                            case0(b, l, v, c, ip, ci_b, out.clone());
                            case1(b, l, v, c, ip, ci_b, out.clone());
                            case2(b, l, v, num_co_b, ip, ci_b, out.clone());
                            case3(b, l, v, num_co_b, ip, ci_b, out.clone());
                        }
                        case0(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                        case1(b, l, v, c, num_ci_b, ci_b_remain, out.clone());
                        case2(b, l, v, num_co_b, num_ci_b, ci_b_remain, out.clone());
                        case3(b, l, v, num_co_b, num_ci_b, ci_b_remain, out.clone());
                    }
                }
            }
        });

        Ok(output)
    }
}
