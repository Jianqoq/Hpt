use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_traits::CommonBounds;
use crate::tensor_base::_Tensor;
use tensor_types::into_scalar::IntoScalar;
use num::traits::MulAdd;
use tensor_types::type_promote::NormalOut;
use tensor_traits::TensorInfo;
use tensor_traits::TensorCreator;
use std::arch::asm;

#[cfg(all(target_feature = "fma", target_feature = "avx2"))]
pub fn conv2d_block_simd_parallel_unroll_i32<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where
        T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>,
        T: IntoScalar<i32>
{
    use wide::i32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
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

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let c_ob = 8;
    let c_ib = 4;
    let w_ob = 14;
    let jp_end = (out_channels + c_ob - 1) / c_ob;
    let ip_end = (in_channels + c_ib - 1) / c_ib;
    let kp_end = (out_width + w_ob - 1) / w_ob;
    (0..jp_end).into_par_iter().for_each_init(
        || output.ptr(),
        |out, jp| {
            let mut res_vectors = [i32x8::splat(0i32); 14];
            let mut res_ptrs = [0 as *mut i32; 14];
            for ip in 0..ip_end {
                for l in 0..out_height {
                    for kp in 0..kp_end {
                        for k in 0..14 {
                            let _k = kp * w_ob + k;
                            let res_ptr = &mut out[jp * c_ob * os2 + _k * os1 + l * os0]; // prettier-ignore
                            let res_vec = unsafe { std::slice::from_raw_parts_mut(res_ptr, 8) }; // prettier-ignore
                            res_vectors[k as usize]
                                .as_array_mut()
                                .copy_from_slice(unsafe {
                                    std::mem::transmute::<&[T], &[i32]>(res_vec)
                                });
                            res_ptrs[k as usize] = res_vec.as_mut_ptr() as *mut i32;
                        }
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                let mut scalar_vec = i32x8::splat(0i32);
                                for i in 0..c_ib {
                                    let _i = ip * c_ib + i;
                                    let kernel_ptr = &kernel[_i * ks2 + jp * c_ob * ks3 + m * ks1 + n * ks0] as *const T; // prettier-ignore
                                    let kernel_vec = unsafe { std::slice::from_raw_parts(kernel_ptr, 8) }; // prettier-ignore
                                    let kernel_vector = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(kernel_vec)) }; // prettier-ignore
                                    let _kernel_vector_arr = kernel_vector.to_array();
                                    for k in 0..14 {
                                        let res_vector = &mut res_vectors[k as usize];

                                        let i_val = inp[_i * is2 + ((kp * w_ob + k) * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                        scalar_vec
                                            .as_array_mut()
                                            .copy_from_slice(&[i_val.into_scalar(); 8]);
                                        let _scalar_arr = scalar_vec.to_array();
                                        let res = kernel_vector * scalar_vec + *res_vector; // prettier-ignore
                                        res_vector
                                            .as_array_mut()
                                            .copy_from_slice(res.as_array_ref());
                                    }
                                }
                            }
                        }
                        for k in 0..14 {
                            let res_vector = &res_vectors[k as usize].as_array_ref();
                            let res_ptr = res_ptrs[k as usize];
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    res_vector.as_ptr() as *const i32,
                                    res_ptr as *mut i32,
                                    8
                                );
                            }
                        }
                    }
                }
            }
        }
    );
    Ok(output)
}

#[cfg(all(target_feature = "fma", target_feature = "avx2"))]
pub fn conv2d_block_simd_parallel_unroll_f32<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where
        T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>,
        T: IntoScalar<f32>
{
    use wide::f32x8;

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
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

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let c_ob = 8;
    let c_ib = 4;
    let w_ob = 14;
    let jp_end = (out_channels + c_ob - 1) / c_ob;
    let ip_end = (in_channels + c_ib - 1) / c_ib;
    let kp_end = (out_width + w_ob - 1) / w_ob;
    (0..jp_end).into_par_iter().for_each_init(
        || output.ptr(),
        |out, jp| {
            let mut res_vectors = [f32x8::splat(0f32); 14];
            let mut res_ptrs = [0 as *mut f32; 14];
            for ip in 0..ip_end {
                for l in 0..out_height {
                    for kp in 0..kp_end {
                        for k in 0..14 {
                            let _k = kp * w_ob + k;
                            let res_ptr = &mut out[jp * c_ob * os2 + _k * os1 + l * os0]; // prettier-ignore
                            let res_vec = unsafe { std::slice::from_raw_parts_mut(res_ptr, 8) }; // prettier-ignore
                            res_vectors[k as usize]
                                .as_array_mut()
                                .copy_from_slice(unsafe {
                                    std::mem::transmute::<&[T], &[f32]>(res_vec)
                                });
                            res_ptrs[k as usize] = res_vec.as_mut_ptr() as *mut f32;
                        }
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                let mut scalar_vec = f32x8::splat(0f32);
                                for i in 0..c_ib {
                                    let _i = ip * c_ib + i;
                                    let kernel_ptr = &kernel[_i * ks2 + jp * c_ob * ks3 + m * ks1 + n * ks0] as *const T; // prettier-ignore
                                    let kernel_vec = unsafe { std::slice::from_raw_parts(kernel_ptr, 8) }; // prettier-ignore
                                    let kernel_vector = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(kernel_vec)) }; // prettier-ignore
                                    let _kernel_vector_arr = kernel_vector.to_array();
                                    for k in 0..14 {
                                        let res_vector = &mut res_vectors[k as usize];

                                        let i_val = inp[_i * is2 + ((kp * w_ob + k) * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                        scalar_vec
                                            .as_array_mut()
                                            .copy_from_slice(&[i_val.into_scalar(); 8]);
                                        let _scalar_arr = scalar_vec.to_array();
                                        let res = kernel_vector * scalar_vec + *res_vector; // prettier-ignore
                                        res_vector
                                            .as_array_mut()
                                            .copy_from_slice(res.as_array_ref());
                                    }
                                }
                            }
                        }
                        for k in 0..14 {
                            let res_vector = &res_vectors[k as usize].as_array_ref();
                            let res_ptr = res_ptrs[k as usize];
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    res_vector.as_ptr() as *const f32,
                                    res_ptr as *mut f32,
                                    8
                                );
                            }
                        }
                    }
                }
            }
        }
    );
    Ok(output)
}

macro_rules! copy_res_ptr_to_registers {
    ($idx:expr, $ymm:expr, $res_ptrs:ident, $out:ident, $indice:expr) => {
        {
            let res_ptr = &mut $out[$indice]; // prettier-ignore
            unsafe { 
                asm!(
                concat!("vmovups ymm", stringify!($ymm), ", [{}]"),
                    in(reg) res_ptr,          
                    options(nostack)
                );
            }
            let res_vec = unsafe { std::slice::from_raw_parts_mut(res_ptr, 8) }; // prettier-ignore
            $res_ptrs[$idx] = res_vec.as_mut_ptr() as *mut f32;
        }
    };
}

macro_rules! perform_fma {
    ($idx:expr, $inp:ident, $indice:expr) => {
        let i_val = $inp[$indice]; // prettier-ignore
        unsafe { 
            asm!(
            "vbroadcastss ymm15, [{}]",
            concat!("vfmadd231ps ymm", stringify!($idx), ", ymm15, ymm14"),
            in(reg) &i_val,
            options(nostack)
        );
        };
    };
}

#[cfg(target_feature = "fma")]
pub fn conv2d_block_simd_parallel_unroll_f32_asm<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where
        T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>,
        T: IntoScalar<f32>
{
    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
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

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let c_ob = 8;
    let w_ob = 14;
    let jp_end = (out_channels + c_ob - 1) / c_ob;
    let kp_end = (out_width + w_ob - 1) / w_ob;
    let mut out = output.ptr();
    (0..jp_end).into_iter().for_each(|jp| {
        // Zero out all the registers
        unsafe {
            asm!(
                "vxorps ymm0, ymm0, ymm0",
                "vxorps ymm1, ymm1, ymm1",
                "vxorps ymm2, ymm2, ymm2",
                "vxorps ymm3, ymm3, ymm3",
                "vxorps ymm4, ymm4, ymm4",
                "vxorps ymm5, ymm5, ymm5",
                "vxorps ymm6, ymm6, ymm6",
                "vxorps ymm7, ymm7, ymm7",
                "vxorps ymm8, ymm8, ymm8",
                "vxorps ymm9, ymm9, ymm9",
                "vxorps ymm10, ymm10, ymm10",
                "vxorps ymm11, ymm11, ymm11",
                "vxorps ymm12, ymm12, ymm12",
                "vxorps ymm13, ymm13, ymm13",
                "vxorps ymm14, ymm14, ymm14",
                "vxorps ymm15, ymm15, ymm15",
                options(nostack)
            );
        }
        let mut res_ptrs = [0 as *mut f32; 14];
        for l in 0..out_height {
            for kp in 0..kp_end {
                copy_res_ptr_to_registers!(0, 0, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 0) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(1, 1, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 1) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(2, 2, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 2) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(3, 3, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 3) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(4, 4, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 4) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(5, 5, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 5) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(6, 6, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 6) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(7, 7, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 7) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(8, 8, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 8) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(9, 9, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 9) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(10, 10, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 10) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(11, 11, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 11) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(12, 12, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 12) * os1 + l * os0); // prettier-ignore
                copy_res_ptr_to_registers!(13, 13, res_ptrs, out, jp * c_ob * os2 + (kp * w_ob + 13) * os1 + l * os0); // prettier-ignore

                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        for i in 0..in_channels {
                            let kernel_ptr = &kernel[i * ks2 + jp * c_ob * ks3 + m * ks1 + n * ks0] as *const T; // prettier-ignore
                            unsafe {
                                asm!(
                                    "vmovups ymm14, [{}]",
                                    in(reg) kernel_ptr,          
                                    options(nostack)
                                );
                            }
                            perform_fma!(0, inp, i * is2 + ((kp * w_ob + 0) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(1, inp, i * is2 + ((kp * w_ob + 1) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(2, inp, i * is2 + ((kp * w_ob + 2) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(3, inp, i * is2 + ((kp * w_ob + 3) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(4, inp, i * is2 + ((kp * w_ob + 4) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(5, inp, i * is2 + ((kp * w_ob + 5) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(6, inp, i * is2 + ((kp * w_ob + 6) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(7, inp, i * is2 + ((kp * w_ob + 7) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(8, inp, i * is2 + ((kp * w_ob + 8) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(9, inp, i * is2 + ((kp * w_ob + 9) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(10, inp, i * is2 + ((kp * w_ob + 10) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(11, inp, i * is2 + ((kp * w_ob + 11) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(12, inp, i * is2 + ((kp * w_ob + 12) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                            perform_fma!(13, inp, i * is2 + ((kp * w_ob + 13) * step_width + m) * is1 + (l * step_height + n) * is0); // prettier-ignore
                        }
                    }
                }
                unsafe {
                    asm!(
                        "vmovups [{}], ymm0",
                        "vmovups [{}], ymm1",
                        "vmovups [{}], ymm2",
                        "vmovups [{}], ymm3",
                        "vmovups [{}], ymm4",
                        "vmovups [{}], ymm5",
                        "vmovups [{}], ymm6",
                        "vmovups [{}], ymm7",
                        "vmovups [{}], ymm8",
                        "vmovups [{}], ymm9",
                        "vmovups [{}], ymm10",
                        "vmovups [{}], ymm11",
                        "vmovups [{}], ymm12",
                        "vmovups [{}], ymm13",
                        in(reg) res_ptrs[0],
                        in(reg) res_ptrs[1],
                        in(reg) res_ptrs[2],
                        in(reg) res_ptrs[3],
                        in(reg) res_ptrs[4],
                        in(reg) res_ptrs[5],
                        in(reg) res_ptrs[6],
                        in(reg) res_ptrs[7],
                        in(reg) res_ptrs[8],
                        in(reg) res_ptrs[9],
                        in(reg) res_ptrs[10],
                        in(reg) res_ptrs[11],
                        in(reg) res_ptrs[12],
                        in(reg) res_ptrs[13],
                        options(nostack)
                    );
                }
            }
        }
    });
    Ok(output)
}
