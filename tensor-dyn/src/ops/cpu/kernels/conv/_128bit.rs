use super::kernels::ConvKernel;
use tensor_common::pointer::Pointer;
use tensor_types::dtype::TypeCommon;
use tensor_types::traits::{Init, VecTrait};
type Vec<T> = <T as TypeCommon>::Vec;

impl ConvKernel for f32 {
    #[rustfmt::skip]
    #[inline]
    fn conv2d_nhwc(
        inp_offset: i64,
        kernel_offset: i64,
        isw_sw: i64,
        inp: &Pointer<f32>,
        kernel: &Pointer<f32>,
        results: &mut [Vec<f32>],
    ) {
        let kernel = unsafe { Vec::<f32>::from_ptr(&kernel[kernel_offset]) };
        let inp0 = Vec::<f32>::splat(inp[inp_offset + isw_sw * 0]);
        let inp1 = Vec::<f32>::splat(inp[inp_offset + isw_sw * 1]);
        let inp2 = Vec::<f32>::splat(inp[inp_offset + isw_sw * 2]);
        let inp3 = Vec::<f32>::splat(inp[inp_offset + isw_sw * 3]);
        unsafe {
            let res0 = inp0.mul_add(kernel, *results.get_unchecked(0));
            let res1 = inp1.mul_add(kernel, *results.get_unchecked(1));
            let res2 = inp2.mul_add(kernel, *results.get_unchecked(2));
            let res3 = inp3.mul_add(kernel, *results.get_unchecked(3));
            results.get_unchecked_mut(0).write_unaligned(res0);
            results.get_unchecked_mut(1).write_unaligned(res1);
            results.get_unchecked_mut(2).write_unaligned(res2);
            results.get_unchecked_mut(3).write_unaligned(res3);
        }
    }

    #[unroll::unroll_for_loops]
    #[rustfmt::skip]
    #[inline]
    fn conv2d_nhwc_store(
        out_offset: i64,
        osw: i64,
        results: &[Vec<f32>],
        out: &mut Pointer<Self>,
    ) {
        for kk in 0..4 {
            let out_vec = &mut out[out_offset + kk as i64 * osw] as *mut _ as *mut Vec<f32>;
            unsafe {
                out_vec.write_unaligned(results[kk]);
            }
        }
    }
}
