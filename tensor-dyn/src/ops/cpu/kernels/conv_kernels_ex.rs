use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;
use tensor_types::{ dtype::TypeCommon, traits::{ Init, VecCommon, VecTrait } };
use tensor_types::type_promote::NormalOut;
use crate::CONV_REGNUM;

#[cfg(not(feature = "bound_check"))]
#[rustfmt::skip]
pub(crate) fn load_store_res_buffer<T, const REGNUM: usize, const LOAD: bool>(
    num_co_rb: i64,
    co_b_remain: i64,
    osw: i64,
    out_offset: i64,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    out: &mut Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecCommon
{
    for j in 0..num_co_rb {
        let buffers = unsafe { res_buffer.get_unchecked_mut(j as usize) };
        for r in 0..REGNUM as i64 {
            unsafe {
                let out_ptr = &mut out[out_offset + j * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64) + r * osw] as *mut _ as *mut T;
                let buffer = buffers.get_unchecked_mut(r as usize) as *mut _ as *mut T;
                if LOAD {
                    std::ptr::copy_nonoverlapping(
                        out_ptr,
                        buffer,
                        <<T as TypeCommon>::Vec as VecCommon>::SIZE
                    );
                } else {
                    std::ptr::copy_nonoverlapping(
                        buffer,
                        out_ptr,
                        <<T as TypeCommon>::Vec as VecCommon>::SIZE
                    );
                }
            }
        }
    }
    let buffers = unsafe { res_buffer.get_unchecked_mut(num_co_rb as usize) };
    for r in 0..REGNUM as i64 {
        unsafe {
            let out_ptr = &mut out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64) + r * osw] as *mut _ as *mut T;
            let buffer = buffers.get_unchecked_mut(r as usize) as *mut _ as *mut T;
            if LOAD {
                std::ptr::copy_nonoverlapping(out_ptr, buffer, co_b_remain as usize);
            } else {
                std::ptr::copy_nonoverlapping(buffer, out_ptr, co_b_remain as usize);
            }
        }
    }
}

#[cfg(feature = "bound_check")]
#[rustfmt::skip]
pub(crate) fn load_store_res_buffer<T, const REGNUM: usize, const LOAD: bool>(
    num_co_rb: i64,
    co_b_remain: i64,
    osw: i64,
    out_offset: i64,
    res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
    out: &mut Pointer<T>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecCommon
{
    for j in 0..num_co_rb {
        let buffers = &mut res_buffer[j as usize];
        for r in 0..REGNUM as i64 {
            unsafe {
                let out_ptr = &mut out[out_offset + j * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64) + r * osw] as *mut _ as *mut T;
                let buffer = &mut buffers[r as usize];
                if LOAD {
                    buffer.copy_from_slice(
                        core::slice::from_raw_parts(out_ptr, <<T as TypeCommon>::Vec as VecCommon>::SIZE)
                    );
                } else {
                    for i in 0..<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64 {
                        out[out_offset + j * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64) + r * osw + i] = buffer.extract(i as usize);
                    }
                }
            }
        }
    }
    let buffers = &mut res_buffer[num_co_rb as usize];
    for r in 0..REGNUM as i64 {
        unsafe {
            let out_ptr = &mut out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64) + r * osw];
            let buffer = &mut buffers[r as usize];
            if LOAD {
                assert!(co_b_remain as usize <= <<T as TypeCommon>::Vec as VecCommon>::SIZE);
                core::ptr::copy_nonoverlapping(out_ptr, buffer.as_mut_ptr(), co_b_remain as usize);
            } else {
                for i in 0..co_b_remain as i64 {
                    out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64) + r * osw + i] = buffer.extract(i as usize);
                }
            }
        }
    }
}

#[rustfmt::skip]
pub(crate) fn pack_kernel<T>(
    num_co_rb: i64,
    co_b_remain: i64,
    kernel_offset: i64,
    kernel: &Pointer<T>,
    kernel_buffer: &mut Vec<<T as TypeCommon>::Vec>
)
    where T: CommonBounds, <T as TypeCommon>::Vec: VecCommon
{
    for j in 0..num_co_rb {
        #[cfg(not(feature = "bound_check"))]
        let kernel_buffer = unsafe { kernel_buffer.get_unchecked_mut(j as usize) };
        #[cfg(feature = "bound_check")]
        let kernel_buffer = &mut kernel_buffer[j as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                &kernel[kernel_offset + j * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64)] as *const _ as *const T,
                kernel_buffer as *mut _ as *mut T,
                <<T as TypeCommon>::Vec as VecCommon>::SIZE
            );
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            &kernel[kernel_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64)] as *const _ as *const T,
            kernel_buffer.get_unchecked_mut(num_co_rb as usize) as *mut _ as *mut T,
            co_b_remain as usize
        );
    }
}

macro_rules! micro_kernel {
    ($num:tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num>]<T, const REGNUM: usize>(
                j: i64,
                kp: i64,
                i: i64,
                inp_offset: i64,
                co_offset: i64,
                kernel_offset: i64,
                step_width: i64,
                isw: i64,
                inp: &Pointer<T>,
                outs: &mut [<T as TypeCommon>::Vec; REGNUM],
                kernel: &Pointer<T>,
            )
                where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecCommon
            {
                $(
                    let _k = kp * (CONV_REGNUM as i64) + $idx;
                    let [<inp_vec $idx>] = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
                )*
                unsafe {
                    let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                        &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64)] as *const _
                    );
                    $(
                        outs[$idx] = [<inp_vec $idx>]._mul_add(kernel_vec, outs[$idx]);
                    )*
                }
            }
        }
    };
}

macro_rules! micro_kernel_dynamic_regnum {
    ($num:tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num _dyn>]<T, const REGNUM: usize>(
                j: i64,
                kp: i64,
                i: i64,
                inp_offset: i64,
                co_offset: i64,
                _: i64,
                kernel_offset: i64,
                step_width: i64,
                isw: i64,
                _: i64,
                inp: &Pointer<T>,
                outs: &mut [<T as TypeCommon>::Vec],
                kernel: &Pointer<T>,
            )
                where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecCommon
            {
                $(
                    let _k = kp * (CONV_REGNUM as i64) + $idx;
                    let [<inp_vec $idx>] = <T as TypeCommon>::Vec::splat(inp[inp_offset + _k * step_width * isw + i]);
                )*
                unsafe {
                    let kernel_vec = <T as TypeCommon>::Vec::from_ptr(
                        &kernel[co_offset + kernel_offset + j * (<<T as TypeCommon>::Vec as VecCommon>::SIZE as i64)] as *const _
                    );
                    $(
                        outs[$idx] = [<inp_vec $idx>]._mul_add(kernel_vec, outs[$idx]);
                    )*
                }
            }
        }
    };
}

macro_rules! micro_kernel_1 {
    ($number:expr, $num:tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num _1>]<T>(
                kp: i64,
                i: i64,
                inp_offset: i64,
                co_offset: i64,
                kernel_offset: i64,
                step_width: i64,
                isw: i64,
                inp: &Pointer<T>,
                out: &mut [T; $number],
                kernel: &Pointer<T>
            )
                where T: CommonBounds + NormalOut<Output = T>, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecCommon
            {
                $(
                    let _k = kp * (CONV_REGNUM as i64) + $idx;
                    let [<inp $idx>] = inp[inp_offset + _k * step_width * isw + i];
                )*
                let kernel_val = kernel[co_offset + kernel_offset];
                $(
                    out[$idx] = [<inp $idx>]._mul(kernel_val)._add(out[$idx]);
                )*
            }
        }
    };
}

macro_rules! micro_kernel_with_buffer {
    ($num:tt, [$($idx:expr),*]) => {
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
                        #[cfg(not(feature = "bound_check"))]
                        let kernel_vec = *kernel.get_unchecked(j as usize);
                        #[cfg(feature = "bound_check")]
                        let kernel_vec = *kernel.get(j as usize).unwrap();
                        #[cfg(not(feature = "bound_check"))]
                        let res_vectors = res_buffer.get_unchecked_mut(j as usize);
                        #[cfg(feature = "bound_check")]
                        let res_vectors = res_buffer.get_mut(j as usize).unwrap();
                        
                        $(
                            #[cfg(not(feature = "bound_check"))]
                            let [<out_vec $idx>] = res_vectors.get_unchecked_mut($idx) as *mut _ as *mut <T as TypeCommon>::Vec;
                            #[cfg(feature = "bound_check")]
                            let [<out_vec $idx>] = res_vectors.get_mut($idx).unwrap() as *mut _ as *mut <T as TypeCommon>::Vec;
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
#[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
micro_kernel!(regnum, [0, 1, 2]);
micro_kernel!(1, [0]);
micro_kernel!(2, [0, 1]);
micro_kernel!(3, [0, 1, 2]);
micro_kernel!(4, [0, 1, 2, 3]);
micro_kernel!(5, [0, 1, 2, 3, 4]);
micro_kernel!(6, [0, 1, 2, 3, 4, 5]);
micro_kernel_dynamic_regnum!(1, [0]);
micro_kernel_dynamic_regnum!(2, [0, 1]);
micro_kernel_dynamic_regnum!(3, [0, 1, 2]);
micro_kernel_dynamic_regnum!(4, [0, 1, 2, 3]);
micro_kernel_dynamic_regnum!(5, [0, 1, 2, 3, 4]);
micro_kernel_dynamic_regnum!(6, [0, 1, 2, 3, 4, 5]);

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
#[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
micro_kernel_with_buffer!(regnum, [0, 1, 2]);
micro_kernel_with_buffer!(1, [0]);
micro_kernel_with_buffer!(2, [0, 1]);
micro_kernel_with_buffer!(3, [0, 1, 2]);
micro_kernel_with_buffer!(4, [0, 1, 2, 3]);
micro_kernel_with_buffer!(5, [0, 1, 2, 3, 4]);
micro_kernel_with_buffer!(6, [0, 1, 2, 3, 4, 5]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(7, [0, 1, 2, 3, 4, 5, 6]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(8, [0, 1, 2, 3, 4, 5, 6, 7]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(9, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_with_buffer!(14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);

#[cfg(target_feature = "avx2")]
micro_kernel_1!(7, regnum, [0, 1, 2, 3, 4, 5, 6]);
#[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
micro_kernel_1!(3, regnum, [0, 1, 2]);
#[cfg(any(target_feature = "avx512f", target_feature = "neon"))]
micro_kernel_1!(15, regnum, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
micro_kernel_1!(1, 1, [0]);
micro_kernel_1!(2, 2, [0, 1]);
micro_kernel_1!(3, 3, [0, 1, 2]);
micro_kernel_1!(4, 4, [0, 1, 2, 3]);
micro_kernel_1!(5, 5, [0, 1, 2, 3, 4]);
micro_kernel_1!(6, 6, [0, 1, 2, 3, 4, 5]);
