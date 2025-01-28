use crate::CommonBounds;
use tensor_common::Pointer;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::traits::VecTrait;
use tensor_types::type_promote::NormalOut;

pub(crate) struct Params {
    pub(crate) arg1: [i64; 2],
    pub(crate) arg2: [i64; 2],
    pub(crate) arg3: [i64; 4],
    pub(crate) arg4: [i64; 3],
    pub(crate) arg5: [i64; 2],
    pub(crate) arg6: [i64; 3],
    pub(crate) pads: [i64; 2],
    pub(crate) arg8: [i64; 2],
    pub(crate) arg9: [i64; 2],
}

pub(crate) struct PartialParams {
    pub(crate) arg1: [i64; 2],
    pub(crate) arg2: [i64; 2],
    pub(crate) arg3: [i64; 4],
    pub(crate) arg4: [i64; 3],
    pub(crate) arg5: [i64; 2],
    pub(crate) arg6: [i64; 3],
    pub(crate) arg7: [i64; 2],
    pub(crate) arg8: [i64; 2],
    pub(crate) arg9: [i64; 2],
    pub(crate) oc_remain: i64,
}

fn template_function<T: CommonBounds>(
    params: Params,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T::Vec) -> T::Vec,
) where
    bool: IntoScalar<T>,
{
    let Params {
        arg1: [oo, o_end],
        arg2: [kh, kw],
        arg3: [b, l, k, i],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    } = params;
    const IC_NVEC: usize = 2;
    const IW_BLOCK: usize = 1;
    for n in 0..kh {
        let h_out = l * step_height + n * dh - ph_start;
        let h_in_range = h_out >= 0 && h_out < out_height;
        if h_in_range {
            for m in 0..kw {
                for o in oo..o_end {
                    let kr0 = unsafe { T::Vec::from_ptr(&kernel[0 * T::Vec::SIZE]) };
                    let kr1 = unsafe { T::Vec::from_ptr(&kernel[1 * T::Vec::SIZE]) };
                    for kk in 0..IW_BLOCK as i64 {
                        let w_out = (k + kk) * step_width + m * dw - pw_start;
                        if w_out >= 0 && w_out < out_width {
                            let out_idx0 =
                                b * isb + h_out * ish + w_out * isw + i + 0 * T::Vec::SIZE as i64;
                            let mut out_vec0 = unsafe { T::Vec::from_ptr(&out[out_idx0]) };
                            let out_idx1 =
                                b * isb + h_out * ish + w_out * isw + i + 1 * T::Vec::SIZE as i64;
                            let mut out_vec1 = unsafe { T::Vec::from_ptr(&out[out_idx1]) };
                            let inp_idx = b * osb + l * osh + (k + kk) * osw + o;
                            let inp_vec = T::Vec::splat(inp[inp_idx]);
                            out_vec0 = inp_vec._mul_add(kr0, out_vec0);
                            out_vec1 = inp_vec._mul_add(kr1, out_vec1);
                            for i in 0..T::Vec::SIZE as i64 {
                                out[out_idx0 + i] = out_vec0[i as usize];
                                out[out_idx1 + i] = out_vec1[i as usize];
                            }
                        }
                    }
                    kernel.add(IC_NVEC * T::Vec::SIZE);
                }
            }
        } else {
            kernel.add(IC_NVEC * T::Vec::SIZE * (kw as usize) * ((o_end - oo) as usize));
        }
    }
}
