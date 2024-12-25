use std::arch::x86_64::*;

use crate::sleef_types::*;

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo32(g: VMask) -> i32 {
        (_mm_movemask_epi8(g) == 0xFFFF) as i32
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo64(g: VMask) -> i32 {
        (_mm_movemask_epi8(g) == 0xFFFF) as i32
}

#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vi(p: *mut i32, v: VInt) {
        _mm_storeu_si128(p as *mut __m128i, v);
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_xor_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_xor_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo64_vm(x: VMask, y: VMask) -> VMask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo32_vm(x: VMask, y: VMask) -> VMask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo32_vo64(m: Vopmask) -> Vopmask {
            _mm_shuffle_epi32(m, 0x08)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo64_vo32(m: Vopmask) -> Vopmask {
            _mm_shuffle_epi32(m, 0x50)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi_vd(vd: VDouble) -> VInt {
        _mm_cvtpd_epi32(vd)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi_vd(vd: VDouble) -> VInt {
        _mm_cvttpd_epi32(vd)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_vi(vi: VInt) -> VDouble {
        _mm_cvtepi32_pd(vi)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi_i(i: i32) -> VInt {
        _mm_set_epi32(0, 0, i, i)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vm_vi(vi: VInt) -> VInt2 {
            _mm_and_si128(_mm_shuffle_epi32(vi, 0x73), _mm_set_epi32(-1, 0, -1, 0))
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vi_vm(vi: VInt2) -> VInt {
            _mm_shuffle_epi32(vi, 0x0d)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vd_vd(vd: VDouble) -> VDouble {
            vcast_vd_vi(vtruncate_vi_vd(vd))
}

#[inline(always)]
pub(crate) unsafe fn vrint_vd_vd(vd: VDouble) -> VDouble {
            vcast_vd_vi(vrint_vi_vd(vd))
}

#[inline(always)]
pub(crate) unsafe fn veq64_vo_vm_vm(x: VMask, y: VMask) -> Vopmask {
            let t = _mm_cmpeq_epi32(x, y);
        vand_vm_vm_vm(t, _mm_shuffle_epi32(t, 0xb1))
}

#[inline(always)]
pub(crate) unsafe fn vadd64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
        _mm_add_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_i_i(i0: i32, i1: i32) -> VMask {
            _mm_set_epi32(i0, i1, i0, i1)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_d(d: f64) -> VDouble {
        _mm_set1_pd(d)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vd(vd: VDouble) -> VMask {
        _mm_castpd_si128(vd)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vd_vm(vm: VMask) -> VDouble {
        _mm_castsi128_pd(vm)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_add_pd(x, y) }

#[inline(always)]
pub(crate) unsafe fn vsub_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_sub_pd(x, y) }

#[inline(always)]
pub(crate) unsafe fn vmul_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_mul_pd(x, y) }

#[inline(always)]
pub(crate) unsafe fn vrec_vd_vd(x: VDouble) -> VDouble {
        _mm_div_pd(_mm_set1_pd(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vd_vd(x: VDouble) -> VDouble {
    _mm_sqrt_pd(x) }

#[inline(always)]
pub(crate) unsafe fn vabs_vd_vd(d: VDouble) -> VDouble {
        _mm_andnot_pd(_mm_set1_pd(-0.0), d)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vd_vd(d: VDouble) -> VDouble {
        _mm_xor_pd(_mm_set1_pd(-0.0), d)
}

#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
        vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
        vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmax_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_max_pd(x, y) }

#[inline(always)]
pub(crate) unsafe fn vmin_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_min_pd(x, y) }

#[inline(always)]
pub(crate) unsafe fn veq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
        _mm_castpd_si128(_mm_cmpeq_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
        _mm_castpd_si128(_mm_cmpneq_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
        _mm_castpd_si128(_mm_cmplt_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
        _mm_castpd_si128(_mm_cmple_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
        _mm_castpd_si128(_mm_cmpgt_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
        _mm_castpd_si128(_mm_cmpge_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi_vi_vi(x: VInt, y: VInt) -> VInt {
        _mm_add_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi_vi_vi(x: VInt, y: VInt) -> VInt {
        _mm_sub_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi_vi(e: VInt) -> VInt {
            vsub_vi_vi_vi(vcast_vi_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vi_vi(x: VInt, y: VInt) -> VInt {
        _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi_vi_vi(x: VInt, y: VInt) -> VInt {
            _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi_vi_vi(x: VInt, y: VInt) -> VInt {
        _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vo_vi(x: Vopmask, y: VInt) -> VInt {
            _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi_vi_i<const C: i32>(x: VInt) -> VInt {
        _mm_slli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi_vi_i<const C: i32>(x: VInt) -> VInt {
        _mm_srli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi_vi_i<const C: i32>(x: VInt) -> VInt {
        _mm_srai_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
        _mm_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
        _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi_vo_vi_vi(m: Vopmask, x: VInt, y: VInt) -> VInt {
            vor_vm_vm_vm(
        vand_vm_vm_vm(m, x),            vandnot_vm_vm_vm(m, y),     )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vd_vd(opmask: Vopmask, x: VDouble, y: VDouble) -> VDouble {
            _mm_or_pd(
        _mm_and_pd(_mm_castsi128_pd(opmask), x),            _mm_andnot_pd(_mm_castsi128_pd(opmask), y),     )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_d_d(o: Vopmask, v1: f64, v0: f64) -> VDouble {
        vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0))
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vd(d: VDouble) -> Vopmask {
        vreinterpret_vm_vd(_mm_cmpeq_pd(vabs_vd_vd(d), _mm_set1_pd(f64::INFINITY)))
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vd(d: VDouble) -> Vopmask {
        vreinterpret_vm_vd(_mm_cmpeq_pd(d, _mm_set1_pd(f64::INFINITY)))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vd(d: VDouble) -> Vopmask {
            vreinterpret_vm_vd(_mm_cmpneq_pd(d, d))
}

#[inline(always)]
pub(crate) unsafe fn vgather_vd_p_vi(ptr: *const f64, vi: VInt) -> VDouble {
        let mut a = [0i32; std::mem::size_of::<VInt>() / std::mem::size_of::<i32>()];
    vstoreu_v_p_vi(a.as_mut_ptr(), vi);
    _mm_set_pd(*ptr.add(a[1] as usize), *ptr.add(a[0] as usize))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_vi2(vi: VInt2) -> VMask {
        vi
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi2_vf(vf: VFloat) -> VInt2 {
        _mm_cvtps_epi32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi2_vf(vf: VFloat) -> VInt2 {
        _mm_cvttps_epi32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vi2(vi: VInt2) -> VFloat {
        _mm_cvtepi32_ps(vcast_vm_vi2(vi))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_f(f: f32) -> VFloat {
        _mm_set1_ps(f)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_i(i: i32) -> VInt2 {
        _mm_set1_epi32(i)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vf(vf: VFloat) -> VMask {
        _mm_castps_si128(vf)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vm(vm: VMask) -> VFloat {
        _mm_castsi128_ps(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vi2(vm: VInt2) -> VFloat {
        _mm_castsi128_ps(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vi2_vf(vf: VFloat) -> VInt2 {
        _mm_castps_si128(vf)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vf_vf(vd: VFloat) -> VFloat {
        vcast_vf_vi2(vtruncate_vi2_vf(vd))
}

#[inline(always)]
pub(crate) unsafe fn vrint_vf_vf(vf: VFloat) -> VFloat {
        vcast_vf_vi2(vrint_vi2_vf(vf))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_add_ps(x, y) }

#[inline(always)]
pub(crate) unsafe fn vsub_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_sub_ps(x, y) }

#[inline(always)]
pub(crate) unsafe fn vmul_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_mul_ps(x, y) }

#[inline(always)]
pub(crate) unsafe fn vdiv_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_div_ps(x, y) }

#[inline(always)]
pub(crate) unsafe fn vrec_vf_vf(x: VFloat) -> VFloat {
        vdiv_vf_vf_vf(vcast_vf_f(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vf_vf(x: VFloat) -> VFloat {
    _mm_sqrt_ps(x) }

#[inline(always)]
pub(crate) unsafe fn vabs_vf_vf(f: VFloat) -> VFloat {
        vreinterpret_vf_vm(vandnot_vm_vm_vm(
        vreinterpret_vm_vf(vcast_vf_f(-0.0)),
        vreinterpret_vm_vf(f),
    ))
}

#[inline(always)]
pub(crate) unsafe fn vneg_vf_vf(d: VFloat) -> VFloat {
        vreinterpret_vf_vm(vxor_vm_vm_vm(
        vreinterpret_vm_vf(vcast_vf_f(-0.0)),
        vreinterpret_vm_vf(d),
    ))
}

#[inline(always)]
pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vmax_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_max_ps(x, y) }

#[inline(always)]
pub(crate) unsafe fn vmin_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_min_ps(x, y) }

#[inline(always)]
pub(crate) unsafe fn veq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
        vreinterpret_vm_vf(_mm_cmpeq_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
        vreinterpret_vm_vf(_mm_cmpneq_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
        vreinterpret_vm_vf(_mm_cmplt_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
        vreinterpret_vm_vf(_mm_cmple_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
        vreinterpret_vm_vf(_mm_cmpgt_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
        vreinterpret_vm_vf(_mm_cmpge_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
        vadd_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
        vsub_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi2_vi2(e: VInt2) -> VInt2 {
        vsub_vi2_vi2_vi2(vcast_vi2_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
        vand_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
        vandnot_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
        vor_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi2_vo_vi2(x: Vopmask, y: VInt2) -> VInt2 {
        vand_vi_vo_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
        vsll_vi_vi_i::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
        vsrl_vi_vi_i::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
        vsra_vi_vi_i::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
        _mm_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
        _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
        _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi2_vo_vi2_vi2(m: Vopmask, x: VInt2, y: VInt2) -> VInt2 {
        vor_vi2_vi2_vi2(
        vand_vi2_vi2_vi2(m, x),            vandnot_vi2_vi2_vi2(m, y),     )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vf_vf(opmask: Vopmask, x: VFloat, y: VFloat) -> VFloat {
        _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(opmask), x),
        _mm_andnot_ps(_mm_castsi128_ps(opmask), y),
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_f_f(o: Vopmask, v1: f32, v0: f32) -> VFloat {
        vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0))
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vf(d: VFloat) -> Vopmask {
        veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vf(d: VFloat) -> Vopmask {
        veq_vo_vf_vf(d, vcast_vf_f(f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vf(d: VFloat) -> Vopmask {
        vneq_vo_vf_vf(d, d)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm_fmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmanp_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm_fnmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm_fmsub_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm_fmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm_fnmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm_fmsub_ps(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vsrl64_vm_vm_i<const C: i32>(x: VMask) -> VMask {
    _mm_srli_epi64::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsub64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_sub_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vf_p_vi2(ptr: *const f32, vi2: VInt2) -> VFloat {
    _mm_i32gather_ps(ptr, vi2, 4)
}
