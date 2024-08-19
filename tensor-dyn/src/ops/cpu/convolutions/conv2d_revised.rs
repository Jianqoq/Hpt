use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_common::slice;
use tensor_types::into_scalar::IntoScalar;
use wide::f32x8;
use tensor_types::dtype::TypeCommon;
use tensor_traits::{ TensorCreator, TensorInfo };
use crate::tensor_base::_Tensor;
use crate::ops::cpu::vector::traits::VecTrait;
use crate::slice::SliceOps;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;

pub fn conv2d_ex_f32_revised<
    const VEC_SIZE: usize,
    const INNER_SIZE: usize,
    const OUTER_SIZE: usize,
    const TOTAL: usize
>(
    img: &_Tensor<f32>,
    kernels: &_Tensor<f32>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
) -> anyhow::Result<_Tensor<f32>> {
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
    let ((pw_start, pw_end), (ph_start, ph_end)) = (padding[0], padding[1]);
    let (dw, dh) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
        let img_padded = _Tensor::<f32>::zeros([
            img_height + ph_start + ph_end,
            img_width + pw_start + pw_end,
            img_channels,
        ])?;
        let he = img_height + ph_start;
        let we = img_width + pw_start;
        let mut slice = slice!(img_padded[ph_start:he, pw_start:we, :])?;
        slice.assign(&img);
        img_padded
    } else {
        img.clone()
    };
    let output = _Tensor::<f32>::zeros([out_height, out_width, out_channels])?;
    let out = output.ptr();
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

    let oc_r8 = out_channels % ((VEC_SIZE * INNER_SIZE) as i64);
    if oc_r8 > 0 {
    } else {
        let ow_r14 = out_width % (OUTER_SIZE as i64);
        if ow_r14 > 0 {
            let jp_end = out_channels / ((VEC_SIZE * INNER_SIZE) as i64);
            let kp_end = out_width / (OUTER_SIZE as i64);
            (0..jp_end).into_par_iter().for_each_init(
                || out,
                |out, jp| {
                    let mut res_vectors = [f32x8::splat(f32::ZERO); TOTAL];
                    let mut res_ptrs = [0 as *mut f32; TOTAL];
                    let mut kernel_vectors = [f32x8::splat(f32::ZERO); INNER_SIZE];
                    for l in 0..out_height {
                        for kp in 0..kp_end {
                            for k in 0..OUTER_SIZE as i64 {
                                let _k = kp * (OUTER_SIZE as i64) + k;
                                for i in 0..INNER_SIZE as i64 {
                                    let res_vec = unsafe { std::slice::from_raw_parts_mut(&mut out[(jp + i) * VEC_SIZE as i64 * os2 + _k * os1 + l * os0], VEC_SIZE) }; // prettier-ignore
                                    res_vectors[
                                        (k * (INNER_SIZE as i64) + i) as usize
                                    ].copy_from_slice(res_vec);
                                    res_ptrs[(k * (INNER_SIZE as i64) + i) as usize] =
                                        res_vec.as_mut_ptr();
                                }
                            }
                            for n in 0..kernel_height {
                                for m in 0..kernel_width {
                                    for i in 0..in_channels {
                                        let kernel_ptr = &kernel[i * ks2 + jp * (INNER_SIZE as i64) * (VEC_SIZE as i64) * ks3 + m * ks1 + n * ks0] as *const f32; // prettier-ignore
                                        kernel_vectors
                                            .iter_mut()
                                            .enumerate()
                                            .for_each(|(o, v)| {
                                                v.copy_from_slice(unsafe {
                                                    std::slice::from_raw_parts(
                                                        kernel_ptr.add(
                                                            ((o as i64) *
                                                                (VEC_SIZE as i64)) as usize
                                                        ),
                                                        VEC_SIZE
                                                    )
                                                });
                                            });
                                        for k in 0..OUTER_SIZE as i64 {
                                            let _k = kp * (OUTER_SIZE as i64) + k;
                                            let scalar = f32x8::splat(inp[i * is2 + (_k * step_width + m * dw) * is1 + (l * step_height + n * dh) * is0].into_scalar()); // prettier-ignore
                                            for o in 0..INNER_SIZE as i64 {
                                                let idx = o * (OUTER_SIZE as i64) + k;
                                                res_vectors[idx as usize].fma(
                                                    kernel_vectors[o as usize],
                                                    scalar
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            for k in 0..TOTAL {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        res_vectors[k].as_ptr(),
                                        res_ptrs[k],
                                        VEC_SIZE
                                    );
                                }
                            }
                        }
                    }
                }
            );
        }
    }
    Ok(output)
}
