use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use wide::f32x8;
use tensor_traits::CommonBounds;
use crate::tensor_base::_Tensor;
use tensor_types::into_scalar::IntoScalar;
use num::traits::MulAdd;
use tensor_types::type_promote::NormalOut;
use tensor_traits::TensorInfo;
use tensor_traits::TensorCreator;

#[cfg(target_feature = "fma")]
pub fn conv2d_block_simd_parallel_unroll<T>(
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
    let c_ib = 4;
    let w_ob = 14;
    let jp_end = (out_channels + c_ob - 1) / c_ob;
    let ip_end = (in_channels + c_ib - 1) / c_ib;
    let kp_end = (out_width + w_ob - 1) / w_ob;
    (0..jp_end).into_par_iter().for_each_init(
        || output.ptr(),
        |out, jp| {
            for ip in 0..ip_end {
                for l in 0..out_height {
                    for kp in 0..kp_end {
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                let k0 = kp * w_ob + 0;
                                let res_ptr0 = &mut out[jp * c_ob * os2 + k0 * os1 + l * os0]; // prettier-ignore
                                let res_vec0 = unsafe { std::slice::from_raw_parts_mut(res_ptr0, 8) }; // prettier-ignore
                                let mut res_vector0 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec0)) }; // prettier-ignore

                                let k1 = kp * w_ob + 1;
                                let res_ptr1 = &mut out[jp * c_ob * os2 + k1 * os1 + l * os0]; // prettier-ignore
                                let res_vec1 = unsafe { std::slice::from_raw_parts_mut(res_ptr1, 8) }; // prettier-ignore
                                let mut res_vector1 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec1)) }; // prettier-ignore

                                let k2 = kp * w_ob + 2;
                                let res_ptr2 = &mut out[jp * c_ob * os2 + k2 * os1 + l * os0]; // prettier-ignore
                                let res_vec2 = unsafe { std::slice::from_raw_parts_mut(res_ptr2, 8) }; // prettier-ignore
                                let mut res_vector2 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec2)) }; // prettier-ignore

                                let k3 = kp * w_ob + 3;
                                let res_ptr3 = &mut out[jp * c_ob * os2 + k3 * os1 + l * os0]; // prettier-ignore
                                let res_vec3 = unsafe { std::slice::from_raw_parts_mut(res_ptr3, 8) }; // prettier-ignore
                                let mut res_vector3 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec3)) }; // prettier-ignore

                                let k4 = kp * w_ob + 4;
                                let res_ptr4 = &mut out[jp * c_ob * os2 + k4 * os1 + l * os0]; // prettier-ignore
                                let res_vec4 = unsafe { std::slice::from_raw_parts_mut(res_ptr4, 8) }; // prettier-ignore
                                let mut res_vector4 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec4)) }; // prettier-ignore

                                let k5 = kp * w_ob + 5;
                                let res_ptr5 = &mut out[jp * c_ob * os2 + k5 * os1 + l * os0]; // prettier-ignore
                                let res_vec5 = unsafe { std::slice::from_raw_parts_mut(res_ptr5, 8) }; // prettier-ignore
                                let mut res_vector5 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec5)) }; // prettier-ignore

                                let k6 = kp * w_ob + 6;
                                let res_ptr6 = &mut out[jp * c_ob * os2 + k6 * os1 + l * os0]; // prettier-ignore
                                let res_vec6 = unsafe { std::slice::from_raw_parts_mut(res_ptr6, 8) }; // prettier-ignore
                                let mut res_vector6 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec6)) }; // prettier-ignore

                                let k7 = kp * w_ob + 7;
                                let res_ptr7 = &mut out[jp * c_ob * os2 + k7 * os1 + l * os0]; // prettier-ignore
                                let res_vec7 = unsafe { std::slice::from_raw_parts_mut(res_ptr7, 8) }; // prettier-ignore
                                let mut res_vector7 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec7)) }; // prettier-ignore

                                let k8 = kp * w_ob + 8;
                                let res_ptr8 = &mut out[jp * c_ob * os2 + k8 * os1 + l * os0]; // prettier-ignore
                                let res_vec8 = unsafe { std::slice::from_raw_parts_mut(res_ptr8, 8) }; // prettier-ignore
                                let mut res_vector8 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec8)) }; // prettier-ignore

                                let k9 = kp * w_ob + 9;
                                let res_ptr9 = &mut out[jp * c_ob * os2 + k9 * os1 + l * os0]; // prettier-ignore
                                let res_vec9 = unsafe { std::slice::from_raw_parts_mut(res_ptr9, 8) }; // prettier-ignore
                                let mut res_vector9 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec9)) }; // prettier-ignore

                                let k10 = kp * w_ob + 10;
                                let res_ptr10 = &mut out[jp * c_ob * os2 + k10 * os1 + l * os0]; // prettier-ignore
                                let res_vec10 = unsafe { std::slice::from_raw_parts_mut(res_ptr10, 8) }; // prettier-ignore
                                let mut res_vector10 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec10)) }; // prettier-ignore

                                let k11 = kp * w_ob + 11;
                                let res_ptr11 = &mut out[jp * c_ob * os2 + k11 * os1 + l * os0]; // prettier-ignore
                                let res_vec11 = unsafe { std::slice::from_raw_parts_mut(res_ptr11, 8) }; // prettier-ignore
                                let mut res_vector11 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec11)) }; // prettier-ignore

                                let k12 = kp * w_ob + 12;
                                let res_ptr12 = &mut out[jp * c_ob * os2 + k12 * os1 + l * os0]; // prettier-ignore
                                let res_vec12 = unsafe { std::slice::from_raw_parts_mut(res_ptr12, 8) }; // prettier-ignore
                                let mut res_vector12 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec12)) }; // prettier-ignore

                                let k13 = kp * w_ob + 13;
                                let res_ptr13 = &mut out[jp * c_ob * os2 + k13 * os1 + l * os0]; // prettier-ignore
                                let res_vec13 = unsafe { std::slice::from_raw_parts_mut(res_ptr13, 8) }; // prettier-ignore
                                let mut res_vector13 = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec13)) }; // prettier-ignore

                                let mut scalar_vec = f32x8::splat(0f32);
                                for i in 0..c_ib {
                                    let i = ip * c_ib + i;
                                    let kernel_ptr = &kernel[i * ks2 + jp * c_ob * ks3 + m * ks1 + n * ks0] as *const T; // prettier-ignore
                                    let kernel_vec = unsafe { std::slice::from_raw_parts(kernel_ptr, 8) }; // prettier-ignore
                                    let kernel_vector = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(kernel_vec)) }; // prettier-ignore
                                    // create img vector
                                    let k = kp * w_ob + 0;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector0); // prettier-ignore
                                    res_vector0
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 1;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector1); // prettier-ignore
                                    res_vector1
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 2;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector2); // prettier-ignore
                                    res_vector2
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 3;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector3); // prettier-ignore
                                    res_vector3
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 4;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector4); // prettier-ignore
                                    res_vector4
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 5;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector5); // prettier-ignore
                                    res_vector5
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 6;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector6); // prettier-ignore
                                    res_vector6
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 7;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector7); // prettier-ignore
                                    res_vector7
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 8;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector8); // prettier-ignore
                                    res_vector8
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 9;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector9); // prettier-ignore
                                    res_vector9
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 10;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector10); // prettier-ignore
                                    res_vector10
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 11;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector11); // prettier-ignore
                                    res_vector11
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 12;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector12); // prettier-ignore
                                    res_vector12
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 13;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector13); // prettier-ignore
                                    res_vector13
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());
                                }
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vector0.to_array().as_ptr() as *const f32,
                                        res_vec0.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector1.to_array().as_ptr() as *const f32,
                                        res_vec1.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector2.to_array().as_ptr() as *const f32,
                                        res_vec2.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector3.to_array().as_ptr() as *const f32,
                                        res_vec3.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector4.to_array().as_ptr() as *const f32,
                                        res_vec4.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector5.to_array().as_ptr() as *const f32,
                                        res_vec5.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector6.to_array().as_ptr() as *const f32,
                                        res_vec6.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector7.to_array().as_ptr() as *const f32,
                                        res_vec7.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector8.to_array().as_ptr() as *const f32,
                                        res_vec8.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector9.to_array().as_ptr() as *const f32,
                                        res_vec9.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector10.to_array().as_ptr() as *const f32,
                                        res_vec10.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector11.to_array().as_ptr() as *const f32,
                                        res_vec11.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector12.to_array().as_ptr() as *const f32,
                                        res_vec12.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector13.to_array().as_ptr() as *const f32,
                                        res_vec13.as_mut_ptr() as *mut f32,
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    );
    Ok(output)
}

#[cfg(target_feature = "fma")]
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
            for ip in 0..ip_end {
                for l in 0..out_height {
                    for kp in 0..kp_end {
                        for n in 0..kernel_height {
                            for m in 0..kernel_width {
                                let k0 = kp * w_ob + 0;
                                let res_ptr0 = &mut out[jp * c_ob * os2 + k0 * os1 + l * os0]; // prettier-ignore
                                let res_vec0 = unsafe { std::slice::from_raw_parts_mut(res_ptr0, 8) }; // prettier-ignore
                                let mut res_vector0 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec0)) }; // prettier-ignore

                                let k1 = kp * w_ob + 1;
                                let res_ptr1 = &mut out[jp * c_ob * os2 + k1 * os1 + l * os0]; // prettier-ignore
                                let res_vec1 = unsafe { std::slice::from_raw_parts_mut(res_ptr1, 8) }; // prettier-ignore
                                let mut res_vector1 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec1)) }; // prettier-ignore

                                let k2 = kp * w_ob + 2;
                                let res_ptr2 = &mut out[jp * c_ob * os2 + k2 * os1 + l * os0]; // prettier-ignore
                                let res_vec2 = unsafe { std::slice::from_raw_parts_mut(res_ptr2, 8) }; // prettier-ignore
                                let mut res_vector2 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec2)) }; // prettier-ignore

                                let k3 = kp * w_ob + 3;
                                let res_ptr3 = &mut out[jp * c_ob * os2 + k3 * os1 + l * os0]; // prettier-ignore
                                let res_vec3 = unsafe { std::slice::from_raw_parts_mut(res_ptr3, 8) }; // prettier-ignore
                                let mut res_vector3 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec3)) }; // prettier-ignore

                                let k4 = kp * w_ob + 4;
                                let res_ptr4 = &mut out[jp * c_ob * os2 + k4 * os1 + l * os0]; // prettier-ignore
                                let res_vec4 = unsafe { std::slice::from_raw_parts_mut(res_ptr4, 8) }; // prettier-ignore
                                let mut res_vector4 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec4)) }; // prettier-ignore

                                let k5 = kp * w_ob + 5;
                                let res_ptr5 = &mut out[jp * c_ob * os2 + k5 * os1 + l * os0]; // prettier-ignore
                                let res_vec5 = unsafe { std::slice::from_raw_parts_mut(res_ptr5, 8) }; // prettier-ignore
                                let mut res_vector5 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec5)) }; // prettier-ignore

                                let k6 = kp * w_ob + 6;
                                let res_ptr6 = &mut out[jp * c_ob * os2 + k6 * os1 + l * os0]; // prettier-ignore
                                let res_vec6 = unsafe { std::slice::from_raw_parts_mut(res_ptr6, 8) }; // prettier-ignore
                                let mut res_vector6 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec6)) }; // prettier-ignore

                                let k7 = kp * w_ob + 7;
                                let res_ptr7 = &mut out[jp * c_ob * os2 + k7 * os1 + l * os0]; // prettier-ignore
                                let res_vec7 = unsafe { std::slice::from_raw_parts_mut(res_ptr7, 8) }; // prettier-ignore
                                let mut res_vector7 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec7)) }; // prettier-ignore

                                let k8 = kp * w_ob + 8;
                                let res_ptr8 = &mut out[jp * c_ob * os2 + k8 * os1 + l * os0]; // prettier-ignore
                                let res_vec8 = unsafe { std::slice::from_raw_parts_mut(res_ptr8, 8) }; // prettier-ignore
                                let mut res_vector8 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec8)) }; // prettier-ignore

                                let k9 = kp * w_ob + 9;
                                let res_ptr9 = &mut out[jp * c_ob * os2 + k9 * os1 + l * os0]; // prettier-ignore
                                let res_vec9 = unsafe { std::slice::from_raw_parts_mut(res_ptr9, 8) }; // prettier-ignore
                                let mut res_vector9 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec9)) }; // prettier-ignore

                                let k10 = kp * w_ob + 10;
                                let res_ptr10 = &mut out[jp * c_ob * os2 + k10 * os1 + l * os0]; // prettier-ignore
                                let res_vec10 = unsafe { std::slice::from_raw_parts_mut(res_ptr10, 8) }; // prettier-ignore
                                let mut res_vector10 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec10)) }; // prettier-ignore

                                let k11 = kp * w_ob + 11;
                                let res_ptr11 = &mut out[jp * c_ob * os2 + k11 * os1 + l * os0]; // prettier-ignore
                                let res_vec11 = unsafe { std::slice::from_raw_parts_mut(res_ptr11, 8) }; // prettier-ignore
                                let mut res_vector11 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec11)) }; // prettier-ignore

                                let k12 = kp * w_ob + 12;
                                let res_ptr12 = &mut out[jp * c_ob * os2 + k12 * os1 + l * os0]; // prettier-ignore
                                let res_vec12 = unsafe { std::slice::from_raw_parts_mut(res_ptr12, 8) }; // prettier-ignore
                                let mut res_vector12 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec12)) }; // prettier-ignore

                                let k13 = kp * w_ob + 13;
                                let res_ptr13 = &mut out[jp * c_ob * os2 + k13 * os1 + l * os0]; // prettier-ignore
                                let res_vec13 = unsafe { std::slice::from_raw_parts_mut(res_ptr13, 8) }; // prettier-ignore
                                let mut res_vector13 = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(res_vec13)) }; // prettier-ignore

                                let mut scalar_vec = i32x8::splat(0i32);
                                for i in 0..c_ib {
                                    let i = ip * c_ib + i;
                                    let kernel_ptr = &kernel[i * ks2 + jp * c_ob * ks3 + m * ks1 + n * ks0] as *const T; // prettier-ignore
                                    let kernel_vec = unsafe { std::slice::from_raw_parts(kernel_ptr, 8) }; // prettier-ignore
                                    let kernel_vector = unsafe { i32x8::from(std::mem::transmute::<&[T], &[i32]>(kernel_vec)) }; // prettier-ignore
                                    // create img vector
                                    let k = kp * w_ob + 0;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec + res_vector0; // prettier-ignore
                                    res_vector0
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 1;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec + res_vector1; // prettier-ignore
                                    res_vector1
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 2;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec + res_vector2; // prettier-ignore
                                    res_vector2
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 3;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector3; // prettier-ignore
                                    res_vector3
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 4;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector4; // prettier-ignore
                                    res_vector4
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 5;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector5; // prettier-ignore
                                    res_vector5
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 6;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector6; // prettier-ignore
                                    res_vector6
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 7;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector7; // prettier-ignore
                                    res_vector7
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 8;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector8; // prettier-ignore
                                    res_vector8
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 9;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector9; // prettier-ignore
                                    res_vector9
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 10;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector10; // prettier-ignore
                                    res_vector10
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 11;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector11; // prettier-ignore
                                    res_vector11
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 12;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector12; // prettier-ignore
                                    res_vector12
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());

                                    let k = kp * w_ob + 13;
                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    scalar_vec
                                        .as_array_mut()
                                        .copy_from_slice(&[i_val.into_scalar(); 8]);
                                    let res = kernel_vector * scalar_vec +  res_vector13; // prettier-ignore
                                    res_vector13
                                        .as_array_mut()
                                        .copy_from_slice(res.to_array().as_slice());
                                }
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vector0.to_array().as_ptr() as *const i32,
                                        res_vec0.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector1.to_array().as_ptr() as *const i32,
                                        res_vec1.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector2.to_array().as_ptr() as *const i32,
                                        res_vec2.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector3.to_array().as_ptr() as *const i32,
                                        res_vec3.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector4.to_array().as_ptr() as *const i32,
                                        res_vec4.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector5.to_array().as_ptr() as *const i32,
                                        res_vec5.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector6.to_array().as_ptr() as *const i32,
                                        res_vec6.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector7.to_array().as_ptr() as *const i32,
                                        res_vec7.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector8.to_array().as_ptr() as *const i32,
                                        res_vec8.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector9.to_array().as_ptr() as *const i32,
                                        res_vec9.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector10.to_array().as_ptr() as *const i32,
                                        res_vec10.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector11.to_array().as_ptr() as *const i32,
                                        res_vec11.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector12.to_array().as_ptr() as *const i32,
                                        res_vec12.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        res_vector13.to_array().as_ptr() as *const i32,
                                        res_vec13.as_mut_ptr() as *mut i32,
                                        8
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    );
    Ok(output)
}
