use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_traits::CommonBounds;
use crate::tensor_base::_Tensor;
use tensor_types::into_scalar::IntoScalar;
use num::traits::MulAdd;
use tensor_types::type_promote::NormalOut;
use tensor_traits::TensorInfo;
use tensor_traits::TensorCreator;

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
                                let mut res_vectors = [i32x8::splat(0i32); 14];
                                let mut res_ptrs = [0 as *mut i32; 14];
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
            }
        }
    );
    Ok(output)
}
