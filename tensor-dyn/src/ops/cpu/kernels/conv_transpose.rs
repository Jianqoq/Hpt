use crate::CommonBounds;
use tensor_common::Pointer;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::traits::VecTrait;

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
        arg1: [ii, i_end],
        arg2: [kh, kw],
        arg3: [b, l, k, j],
        arg4: [osb, osh, osw],
        arg5: [step_height, step_width],
        arg6: [isb, ish, isw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [img_height, img_width],
    } = params;
    const OW_BLOCK: usize = 1;
    const OC_BLOCK: usize = 1;
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_BLOCK];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..OC_BLOCK {
                ret[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(
                        &out[b * osb
                            + l * osh
                            + (k + kk) * osw
                            + j
                            + v as i64 * T::Vec::SIZE as i64] as *const _
                            as *const T,
                    )
                }; // prettier-ignore
            }
        }
        ret
    };
    let is0 =
        b * isb + l * step_height * ish + k * step_width * isw - pw_start * isw - ph_start * ish;
    let m_must_in_range = k * step_width >= pw_start
        && (k + (OW_BLOCK as i64)) * step_width + (kw - 1) * dw < img_width + pw_start;
    for n in 0..kh {
        let l_in_range = l * step_height + n * dh >= ph_start
            && l * step_height + n * dh < img_height + ph_start;
        let is1 = is0 + n * dh * ish;
        if l_in_range {
            if m_must_in_range {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in ii..i_end {
                        let is3 = is2 + i;
                        kernel.add(OC_BLOCK * T::Vec::SIZE);
                    }
                }
            } else {
                for m in 0..kw {
                    let is2 = is1 + m * dw * isw;
                    for i in ii..i_end {
                        let is3 = is2 + i;
                        kernel.add(OC_BLOCK * T::Vec::SIZE);
                    }
                }
            }
        } else {
            kernel.add(OC_BLOCK * T::Vec::SIZE * (kw as usize) * ((i_end - ii) as usize));
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..OC_BLOCK {
            let out_vec = &mut out
                [b * osb + l * osh + (k + kk) * osw + j + (v * T::Vec::SIZE) as i64]
                as *mut _ as *mut T::Vec; // prettier-ignore
            unsafe {
                out_vec.write_unaligned(activation(results[v as usize][kk as usize]));
            }
        }
    }
}
