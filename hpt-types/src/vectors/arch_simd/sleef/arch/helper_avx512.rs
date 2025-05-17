
use std::arch::x86_64::*;
use crate::sleef_types::{VDouble, VFloat, VInt, VInt2, VMask, Vopmask};

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vd(vd: VDouble) -> VMask {
    _mm512_castpd_si512(vd)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vd_vm(vm: VMask) -> VDouble {
    _mm512_castsi512_pd(vm)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm512_xor_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm512_or_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm512_andnot_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm512_sub_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmul_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm512_mul_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_vi(vi: VInt) -> VDouble {
    _mm512_cvtepi32_pd(vi)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi_vd(vd: VDouble) -> VInt {
    _mm512_cvt_roundpd_epi32::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(vd)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vd_vd(mask: Vopmask, x: VDouble, y: VDouble) -> VDouble {
    _mm512_mask_blend_pd(mask as __mmask8, y, x)
}

#[inline(always)]
pub(crate) unsafe fn veq64_vo_vm_vm(x: VMask, y: VMask) -> Vopmask {
    _mm512_cmp_epi64_mask::<_MM_CMPINT_EQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_d(d: f64) -> VDouble {
    _mm512_set1_pd(d)
}

#[inline(always)]
pub(crate) unsafe fn vor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm512_kor(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vabs_vd_vd(d: VDouble) -> VDouble {
    vreinterpret_vd_vm(_mm512_andnot_si512(vreinterpret_vm_vd(_mm512_set1_pd(-0.0)), vreinterpret_vm_vd(d)))
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vd(d: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(vabs_vd_vd(d), _mm512_set1_pd(f64::INFINITY)) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_GE_OQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vadd_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm512_add_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm512_kand(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_LE_OQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_LT_OQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_GT_OQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(x, y) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm256_and_si256(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi_i(i: i32) -> VInt {
    _mm256_set1_epi32(i)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vi_vm(vi: VMask) -> VInt {
    _mm512_castsi512_si256(_mm512_maskz_permutexvar_epi32(0x00ff, _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1), vi))
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm512_and_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi_vi_i<const C: i32>(x: VInt) -> VInt {
    _mm256_srli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi_vi_i<const C: i32>(x: VInt) -> VInt {
    _mm256_slli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi_vi_i<const C: i32>(x: VInt) -> VInt {
    _mm256_srai_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vadd64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm512_add_epi64(x, y)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm512_fmadd_pd(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm512_fmadd_pd(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vfmanp_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm512_fnmadd_pd(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vfmapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm512_fmsub_pd(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vrec_vd_vd(x: VDouble) -> VDouble {
    _mm512_div_pd(_mm512_set1_pd(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm256_sub_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm256_add_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vm_vi(vi: VInt) -> VMask {
    _mm512_maskz_permutexvar_epi32(0xaaaa, _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), _mm512_castsi256_si512(vi))
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vf(vf: VFloat) -> VMask {
    _mm512_castps_si512(vf)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vm(vm: VMask) -> VFloat {
    _mm512_castsi512_ps(vm)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_f(f: f32) -> VFloat {
    _mm512_set1_ps(f)
}

#[inline(always)]
pub(crate) unsafe fn vabs_vf_vf(f: VFloat) -> VFloat {
    vreinterpret_vf_vm(vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(f)))
}

#[inline(always)]
pub(crate) unsafe fn vneg_vd_vd(d: VDouble) -> VDouble {
    vreinterpret_vd_vm(_mm512_xor_si512(vreinterpret_vm_vd(_mm512_set1_pd(-0.0)), vreinterpret_vm_vd(d)))
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vd_vd(x: VDouble) -> VDouble {
    _mm512_sqrt_pd(x)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm512_add_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm512_and_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_i(i: i32) -> VInt2 {
    _mm512_set1_epi32(i)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm512_fmadd_ps(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmul_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm512_mul_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vf_vf(d: VFloat) -> VFloat {
    vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(d)))
}

#[inline(always)]
pub(crate) unsafe fn vdiv_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm512_div_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vrec_vf_vf(x: VFloat) -> VFloat {
    vdiv_vf_vf_vf(vcast_vf_f(1.0f32), x)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vi2(vi: VInt2) -> VFloat {
    _mm512_castsi512_ps(vi)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vi2_vf(vi: VFloat) -> VInt2 {
    _mm512_castps_si512(vi)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vf_vf(m: Vopmask, x: VFloat, y: VFloat) -> VFloat {
    _mm512_mask_blend_ps(m, y, x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vf_vf(x: VFloat) -> VFloat {
    _mm512_sqrt_ps(x)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm512_sub_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_f_f(o: Vopmask, v1: f32, v0: f32) -> VFloat {
    vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0))
}

#[inline(always)]
pub(crate) unsafe fn vfma_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm512_fmadd_ps(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vfmanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm512_fnmadd_ps(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vfmapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm512_fmsub_ps(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vo_vi(o: Vopmask, y: VInt) -> VInt {
    _mm512_castsi512_si256(_mm512_mask_and_epi32(
        _mm512_set1_epi32(0),
        o,
        _mm512_castsi256_si512(y),
        _mm512_castsi256_si512(y)
    ))
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vo64_vm(o: Vopmask, m: VMask) -> VMask {
    _mm512_mask_and_epi64(
        _mm512_set1_epi32(0),
        o as __mmask8,
        m,
        m
    )
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo64_vm(o: Vopmask, m: VMask) -> VMask {
    _mm512_mask_and_epi64(
        m,
        o as __mmask8,
        _mm512_set1_epi32(0),
        _mm512_set1_epi32(0)
    )
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo64_vm(o: Vopmask, m: VMask) -> VMask {
    _mm512_mask_or_epi64(
        m,
        o as __mmask8,
        _mm512_set1_epi32(-1),
        _mm512_set1_epi32(-1)
    )
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vo32_vm(o: Vopmask, m: VMask) -> VMask {
    _mm512_mask_and_epi32(
        _mm512_set1_epi32(0),
        o,
        m,
        m
    )
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo32_vm(o: Vopmask, m: VMask) -> VMask {
    _mm512_mask_and_epi32(
        m,
        o,
        _mm512_set1_epi32(0),
        _mm512_set1_epi32(0)
    )
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo32_vm(o: Vopmask, m: VMask) -> VMask {
    _mm512_mask_or_epi32(
        m,
        o,
        _mm512_set1_epi32(-1),
        _mm512_set1_epi32(-1)
    )
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm256_andnot_si256(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm512_kandn(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm512_kxor(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo32_vo64(o: Vopmask) -> Vopmask {
    o
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo64_vo32(o: Vopmask) -> Vopmask {
    o
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_i_i(i0: i32, i1: i32) -> VMask {
    _mm512_set_epi32(i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    _mm512_cmp_epi32_mask::<_MM_CMPINT_EQ>(
        _mm512_castsi256_si512(x),
        _mm512_castsi256_si512(y),
    )
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    _mm512_cmp_epi32_mask::<_MM_CMPINT_LT>(
        _mm512_castsi256_si512(y),
        _mm512_castsi256_si512(x),
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_d_d(o: Vopmask, v1: f64, v0: f64) -> VDouble {
    vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0))
}

#[inline(always)]
pub(crate) unsafe fn vgather_vd_p_vi(ptr: *const f64, vi: VInt) -> VDouble {
    _mm512_i32gather_pd::<8>(vi, ptr)
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vd(d: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(d, d) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vd(d: VDouble) -> Vopmask {
    _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(d, _mm512_set1_pd(f64::INFINITY)) as Vopmask
}

#[inline(always)]
pub(crate) unsafe fn vmax_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm512_max_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmin_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm512_min_pd(x, y)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm512_fmsub_pd(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi_vi(e: VInt) -> VInt {
    vsub_vi_vi_vi(vcast_vi_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vd_vd(vd: VDouble) -> VDouble {
    _mm512_roundscale_pd::<{_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC}>(vd)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vd_vd(vd: VDouble) -> VDouble {
    _mm512_roundscale_pd::<{_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC}>(vd)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi_vd(vd: VDouble) -> VInt {
    _mm512_cvt_roundpd_epi32::<{_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC}>(vd)
}

#[inline(always)]
pub(crate) unsafe fn vsrl64_vm_vm_i<const C: u32>(x: VMask) -> VMask {
    _mm512_srli_epi64::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsub64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm512_sub_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi_vo_vi_vi(m: Vopmask, x: VInt, y: VInt) -> VInt {
    _mm512_castsi512_si256(
        _mm512_mask_blend_epi32(
            m,
            _mm512_castsi256_si512(y),
            _mm512_castsi256_si512(x)
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo64(g: Vopmask) -> i32 {
    (g == 0xff) as i32
}

#[inline(always)]
pub(crate) unsafe fn vgetexp_vd_vd(d: VDouble) -> VDouble {
    _mm512_getexp_pd(d)
}

#[inline(always)]
pub(crate) unsafe fn vgetmant_vd_vd(d: VDouble) -> VDouble {
    _mm512_getmant_pd::<0x3, 0x2>(d)
}

#[inline(always)]
pub(crate) unsafe fn vfixup_vd_vd_vd_vi2_i<const IMM: i32>(a: VDouble, b: VDouble, c: VInt2) -> VDouble {
    _mm512_fixupimm_pd::<IMM>(a, b, c)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm512_add_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi2_vo_vi2(o: Vopmask, m: VInt2) -> VInt2 {
    _mm512_mask_and_epi32(
        _mm512_set1_epi32(0),
        o,
        m,
        m
    )
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm512_andnot_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_vi2(vi: VInt2) -> VMask {
    vi
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vi2(vi: VInt2) -> VFloat {
    _mm512_cvtepi32_ps(vcast_vm_vi2(vi))
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(x, y)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    _mm512_cmpeq_epi32_mask(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vf_p_vi2(ptr: *const f32, vi2: VInt2) -> VFloat {
    _mm512_i32gather_ps::<4>(vi2, ptr)
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    _mm512_cmp_ps_mask::<_CMP_GE_OQ>(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    let m = _mm512_cmp_epi32_mask::<_MM_CMPINT_LT>(y, x);
    _mm512_mask_and_epi32(
        _mm512_set1_epi32(0),
        m,
        _mm512_set1_epi32(-1),
        _mm512_set1_epi32(-1)
    )
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    _mm512_cmpgt_epi32_mask(x, y)
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vf(d: VFloat) -> Vopmask {
    veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vf(d: VFloat) -> Vopmask {
    vneq_vo_vf_vf(d, d)
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vf(d: VFloat) -> Vopmask {
    veq_vo_vf_vf(d, vcast_vf_f(f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    _mm512_cmp_ps_mask::<_CMP_LE_OQ>(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmax_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm512_max_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmin_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm512_min_ps(x, y)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm512_fnmadd_ps(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm512_sub_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_vm(vm: VMask) -> VInt2 {
    vm
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi2_vi2(e: VInt2) -> VInt2 {
    vsub_vi2_vi2_vi2(vcast_vi2_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm512_or_si512(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vf_vf(vd: VFloat) -> VFloat {
    _mm512_roundscale_ps::<{_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC}>(vd)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi2_vf(vf: VFloat) -> VInt2 {
    vcast_vi2_vm(_mm512_cvtps_epi32(vf))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi2_vo_vi2_vi2(m: Vopmask, x: VInt2, y: VInt2) -> VInt2 {
    _mm512_mask_blend_epi32(m, y, x)
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi2_vi2_i<const C: u32>(x: VInt2) -> VInt2 {
    _mm512_slli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi2_vi2_i<const C: u32>(x: VInt2) -> VInt2 {
    _mm512_srai_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi2_vi2_i<const C: u32>(x: VInt2) -> VInt2 {
    _mm512_srli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo32(g: Vopmask) -> i32 {
    (g == 0xffff) as i32
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vf_vf(vd: VFloat) -> VFloat {
    _mm512_roundscale_ps::<{_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC}>(vd)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi2_vf(vf: VFloat) -> VInt2 {
    vcast_vi2_vm(_mm512_cvttps_epi32(vf))
}

#[inline(always)]
pub(crate) unsafe fn vgetexp_vf_vf(d: VFloat) -> VFloat {
    _mm512_getexp_ps(d)
}

#[inline(always)]
pub(crate) unsafe fn vgetmant_vf_vf(d: VFloat) -> VFloat {
    _mm512_getmant_ps::<0x3, 0x2>(d)
}

#[inline(always)]
pub(crate) unsafe fn vfixup_vf_vf_vf_vi2_i<const IMM: i32>(a: VFloat, b: VFloat, c: VInt2) -> VFloat {
    _mm512_fixupimm_ps::<IMM>(a, b, c)
}