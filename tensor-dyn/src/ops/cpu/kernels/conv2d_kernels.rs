use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;
use tensor_types::{ dtype::TypeCommon, traits::{ Init, VecSize, VecTrait } };

use crate::CONV_REGNUM;

macro_rules! micro_kernel {
    ($num: tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num>]<T>(
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
                $(
                    let _k = kp * (CONV_REGNUM as i64) + $idx;
                    let [<inp_vec $idx>] = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
                )*
                for j in 0..num_co_rb {
                    let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
                    unsafe {
                        let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                            &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _
                        );
                        $(
                            let [<out_vec $idx>] = &mut out[co_offset + ofs + $idx * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
                        )*
                        $(
                            let [<res $idx>] = [<inp_vec $idx>]._mul_add(kernel_vec, [<out_vec $idx>].read_unaligned());
                        )*
                        $(
                            [<out_vec $idx>].write_unaligned([<res $idx>]);
                        )*
                    }
                }
            }
        }
    };
}

macro_rules! micro_kernel_with_buffer {
    ($num: tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num _with_buffer>]<T>(
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
                $(
                    let _k = kp * (CONV_REGNUM as i64) + $idx;
                    let [<inp_vec $idx>] = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
                )*
                for j in 0..num_co_rb + 1 {
                    unsafe {
                        let kernel_vec = *kernel.get_unchecked(j as usize);
                        let res_vectors = res_buffer.get_unchecked_mut(j as usize);
                        
                        $(
                            let [<out_vec $idx>] = res_vectors.get_unchecked_mut($idx) as *mut _ as *mut <T as TypeCommon>::Vec;
                        )*
                        $(
                            let [<res $idx>] = [<inp_vec $idx>]._mul_add(kernel_vec, [<out_vec $idx>].read_unaligned());
                        )*
                        $(
                            [<out_vec $idx>].write_unaligned([<res $idx>]);
                        )*
                    }
                }
            }
        }
    };
}

#[cfg(target_feature = "avx2")]
micro_kernel!(regnum, [0, 1, 2, 3, 4, 5, 6]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(regnum, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
#[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
micro_kernel!(regnum, [0, 1, 2]);
micro_kernel!(1, [0]);
micro_kernel!(2, [0, 1]);
micro_kernel!(3, [0, 1, 2]);
micro_kernel!(4, [0, 1, 2, 3]);
micro_kernel!(5, [0, 1, 2, 3, 4]);
micro_kernel!(6, [0, 1, 2, 3, 4, 5]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(7, [0, 1, 2, 3, 4, 5, 6]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(8, [0, 1, 2, 3, 4, 5, 6, 7]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(9, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
#[cfg(target_feature = "avx512f")]
micro_kernel!(14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);

#[cfg(target_feature = "avx2")]
micro_kernel_with_buffer!(regnum, [0, 1, 2, 3, 4, 5, 6]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(regnum, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
#[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
micro_kernel_with_buffer!(regnum, [0, 1, 2]);
micro_kernel_with_buffer!(1, [0]);
micro_kernel_with_buffer!(2, [0, 1]);
micro_kernel_with_buffer!(3, [0, 1, 2]);
micro_kernel_with_buffer!(4, [0, 1, 2, 3]);
micro_kernel_with_buffer!(5, [0, 1, 2, 3, 4]);
micro_kernel_with_buffer!(6, [0, 1, 2, 3, 4, 5]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(7, [0, 1, 2, 3, 4, 5, 6]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(8, [0, 1, 2, 3, 4, 5, 6, 7]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(9, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
#[cfg(target_feature = "avx512f")]
micro_kernel_with_buffer!(14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);