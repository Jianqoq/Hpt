use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;
use tensor_types::{ dtype::TypeCommon, traits::{ Init, VecSize, VecTrait } };
use tensor_types::type_promote::NormalOut;
use crate::CONV_REGNUM;

macro_rules! _maxpool {
    () => {
        for l in 0..out_height {
            for n in 0..kernel_height {
                for m in 0..kernel_width {
                    for k in 0..out_width {
                        for j in 0..out_channels {
                            out[l][k][j] = out[l][k][j].max(inp[l * stride_height + n][k * stride_width + m][j]);
                        }
                    }
                }
            }
        }
    };
}

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
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    for j in 0..num_co_rb {
        let buffers = unsafe { res_buffer.get_unchecked_mut(j as usize) };
        for r in 0..REGNUM as i64 {
            unsafe {
                let out_ptr = &mut out[out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw] as *mut _ as *mut T;
                let buffer = buffers.get_unchecked_mut(r as usize) as *mut _ as *mut T;
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
            let out_ptr = &mut out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw] as *mut _ as *mut T;
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
    where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    for j in 0..num_co_rb {
        let buffers = &mut res_buffer[j as usize];
        for r in 0..REGNUM as i64 {
            unsafe {
                let out_ptr = &mut out[out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw] as *mut _ as *mut T;
                let buffer = &mut buffers[r as usize];
                if LOAD {
                    buffer.copy_from_slice(
                        core::slice::from_raw_parts(out_ptr, <<T as TypeCommon>::Vec as VecSize>::SIZE)
                    );
                } else {
                    for i in 0..<<T as TypeCommon>::Vec as VecSize>::SIZE as i64 {
                        out[out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw + i] = buffer.extract(i as usize);
                    }
                }
            }
        }
    }
    let buffers = &mut res_buffer[num_co_rb as usize];
    for r in 0..REGNUM as i64 {
        unsafe {
            let out_ptr = &mut out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw];
            let buffer = &mut buffers[r as usize];
            if LOAD {
                assert!(co_b_remain as usize <= <<T as TypeCommon>::Vec as VecSize>::SIZE);
                core::ptr::copy_nonoverlapping(out_ptr, buffer.as_mut_ptr(), co_b_remain as usize);
            } else {
                for i in 0..co_b_remain as i64 {
                    out[out_offset + num_co_rb * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) + r * osw + i] = buffer.extract(i as usize);
                }
            }
        }
    }
}

macro_rules! micro_kernel {
    ($num:tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num>]<T>(
                num_co_rb: i64,
                kp: i64,
                inp_offset: i64,
                co_offset: i64,
                out_offset: i64,
                step_width: i64,
                isw: i64,
                osw: i64,
                inp: &Pointer<T>,
                out: &mut Pointer<T>,
            )
                where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize + NormalOut<Output=<T as TypeCommon>::Vec>
            {
                for j in 0..num_co_rb {
                    let ofs = out_offset + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
                    unsafe {
                        $(
                            let _k = kp * (CONV_REGNUM as i64) + $idx;
                            let [<inp_vec $idx>] = &inp[inp_offset + _k * step_width * isw + j * (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)] as *const _ as *const <T as TypeCommon>::Vec;
                        )*
                        $(
                            let [<out_vec $idx>] = &mut out[co_offset + ofs + $idx * osw] as *mut _ as *mut <T as TypeCommon>::Vec;
                        )*
                        $(
                            let [<res $idx>] = [<inp_vec $idx>].read_unaligned()._max([<out_vec $idx>].read_unaligned());
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

macro_rules! micro_kernel_1 {
    ($num:tt, [$($idx:expr),*]) => {
        paste::paste! {
            #[rustfmt::skip]
            pub(crate) fn [<micro_kernel_ $num _1>]<T>(
                _: i64,
                kp: i64,
                inp_offset: i64,
                co_offset: i64,
                out_offset: i64,
                step_width: i64,
                isw: i64,
                osw: i64,
                inp: &Pointer<T>,
                out: &mut Pointer<T>,
            )
                where T: CommonBounds + NormalOut<Output = T>, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize + NormalOut<Output = <T as TypeCommon>::Vec>
            {
                $(
                    let _k = kp * (CONV_REGNUM as i64) + $idx;
                    let [<inp_vec $idx>] = inp[inp_offset + _k * step_width * isw];
                )*
                $(
                    out[co_offset + out_offset + $idx * osw] = [<inp_vec $idx>]._max(out[co_offset + out_offset + $idx * osw]);
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
                inp_offset: i64,
                step_width: i64,
                isw: i64,
                inp: &Pointer<T>,
                res_buffer: &mut Vec<Vec<<T as TypeCommon>::Vec>>,
            )
                where T: CommonBounds, <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize + NormalOut<Output=<T as TypeCommon>::Vec>
            {
                for j in 0..num_co_rb + 1 {
                    unsafe {
                        $(
                            let _k = kp * (CONV_REGNUM as i64) + $idx;
                            let [<inp_vec $idx>] = &inp[inp_offset + _k * step_width * isw] as *const _ as *const <T as TypeCommon>::Vec;
                        )*
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
                            let [<res $idx>] = [<inp_vec $idx>].read_unaligned()._max([<out_vec $idx>].read_unaligned());
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

#[rustfmt::skip]
pub(crate)fn micro_kernel_regnum<T>(num_co_rb:i64,kp:i64,inp_offset:i64,co_offset:i64,out_offset:i64,step_width:i64,isw:i64,osw:i64,inp: &Pointer<T>,out: &mut Pointer<T>,)where T:CommonBounds, <T as TypeCommon>::Vec:VecTrait<T> +Copy+Init<T> +VecSize+NormalOut<Output =  <T as TypeCommon>::Vec>{
    for j in 0..num_co_rb {
        let ofs = out_offset+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64);
        unsafe {
            let _k = kp*(CONV_REGNUM as i64)+0;
            let inp_vec0 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let _k = kp*(CONV_REGNUM as i64)+1;
            let inp_vec1 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let _k = kp*(CONV_REGNUM as i64)+2;
            let inp_vec2 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let _k = kp*(CONV_REGNUM as i64)+3;
            let inp_vec3 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let _k = kp*(CONV_REGNUM as i64)+4;
            let inp_vec4 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let _k = kp*(CONV_REGNUM as i64)+5;
            let inp_vec5 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let _k = kp*(CONV_REGNUM as i64)+6;
            let inp_vec6 =  &inp[inp_offset+_k*step_width*isw+j*(<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)]as *const _ as *const <T as TypeCommon>::Vec;
            let out_vec0 =  &mut out[co_offset+ofs+0*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec1 =  &mut out[co_offset+ofs+1*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec2 =  &mut out[co_offset+ofs+2*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec3 =  &mut out[co_offset+ofs+3*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec4 =  &mut out[co_offset+ofs+4*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec5 =  &mut out[co_offset+ofs+5*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let out_vec6 =  &mut out[co_offset+ofs+6*osw]as *mut _ as *mut <T as TypeCommon>::Vec;
            let res0 = inp_vec0.read_unaligned()._max(out_vec0.read_unaligned());
            let res1 = inp_vec1.read_unaligned()._max(out_vec1.read_unaligned());
            let res2 = inp_vec2.read_unaligned()._max(out_vec2.read_unaligned());
            let res3 = inp_vec3.read_unaligned()._max(out_vec3.read_unaligned());
            let res4 = inp_vec4.read_unaligned()._max(out_vec4.read_unaligned());
            let res5 = inp_vec5.read_unaligned()._max(out_vec5.read_unaligned());
            let res6 = inp_vec6.read_unaligned()._max(out_vec6.read_unaligned());
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

micro_kernel_1!(regnum, [0, 1, 2, 3, 4, 5, 6]);
micro_kernel_1!(1, [0]);
micro_kernel_1!(2, [0, 1]);
micro_kernel_1!(3, [0, 1, 2]);
micro_kernel_1!(4, [0, 1, 2, 3]);
micro_kernel_1!(5, [0, 1, 2, 3, 4]);
micro_kernel_1!(6, [0, 1, 2, 3, 4, 5]);
