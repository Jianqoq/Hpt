#![allow(unused)]

use std::mem::transmute;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::sleef_types::*;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Vquad {
    pub(crate) x: VMask,
    pub(crate) y: VMask,
}
impl Vquad {
    fn as_ptr(&self) -> *const u8 {
        unsafe { transmute(self) }
    }
}
type Vargquad = Vquad;

#[inline(always)]
pub(crate) unsafe fn vprefetch_v_p(ptr: *const std::ffi::c_void) {
    _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo32(g: Vopmask) -> i32 {
    _mm_test_all_ones(_mm_and_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1)))
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo64(g: Vopmask) -> i32 {
    _mm_test_all_ones(_mm_and_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1)))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_d(d: f64) -> VDouble {
    _mm256_set1_pd(d)
}
#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vd(d: __m256d) -> VMask {
    _mm256_castpd_si256(d)
}
#[inline(always)]
pub(crate) unsafe fn vreinterpret_vd_vm(m: VMask) -> VDouble {
    _mm256_castsi256_pd(m)
}

#[inline(always)]
pub(crate) unsafe fn vloadu_vi2_p(p: *const i32) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}
#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vi2(p: *mut i32, v: __m256i) {
    _mm256_storeu_si256(p as *mut __m256i, v)
}
#[inline(always)]
pub(crate) unsafe fn vloadu_vi_p(p: *const i32) -> __m128i {
    _mm_loadu_si128(p as *const __m128i)
}
#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vi(p: *mut i32, v: __m128i) {
    _mm_storeu_si128(p as *mut __m128i, v)
}


#[inline(always)]
pub(crate) unsafe fn vand_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vand_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vxor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vand_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo32_vo64(o: Vopmask) -> Vopmask {
    _mm256_permutevar8x32_epi32(o, _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo64_vo32(o: Vopmask) -> Vopmask {
    _mm256_permutevar8x32_epi32(o, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo_i(i: i32) -> Vopmask {
    _mm256_set1_epi64x(if i != 0 { -1 } else { 0 })
}


#[inline(always)]
pub(crate) unsafe fn vrint_vi_vd(vd: VDouble) -> VInt {
    _mm256_cvtpd_epi32(vd)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi_vd(vd: VDouble) -> VInt {
    _mm256_cvttpd_epi32(vd)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vd_vd(vd: VDouble) -> VDouble {
    _mm256_round_pd(vd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vf_vf(vd: VFloat) -> VFloat {
    _mm256_round_ps(vd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vd_vd(vd: VDouble) -> VDouble {
    _mm256_round_pd(vd, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vf_vf(vf: VFloat) -> VFloat {
    _mm256_round_ps(vf, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_vi(vi: VInt) -> VDouble {
    _mm256_cvtepi32_pd(vi)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi_i(i: i32) -> VInt {
    _mm_set1_epi32(i)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vm_vi(vi: VInt) -> VMask {
    _mm256_slli_epi64(_mm256_cvtepi32_epi64(vi), 32)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vi_vm(vi: VMask) -> VInt {
    _mm_or_si128(
        _mm_castps_si128(
            _mm_shuffle_ps(_mm_castsi128_ps(_mm256_castsi256_si128(vi)), _mm_set1_ps(0.0), 0x0d)
        ),
        _mm_castps_si128(
            _mm_shuffle_ps(
                _mm_set1_ps(0.0),
                _mm_castsi128_ps(_mm256_extractf128_si256(vi, 1)),
                0xd0
            )
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_i_i(i0: i32, i1: i32) -> VMask {
    _mm256_set_epi32(i0, i1, i0, i1, i0, i1, i0, i1)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_i64(i: i64) -> VMask {
    _mm256_set1_epi64x(i)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_u64(i: u64) -> VMask {
    _mm256_set1_epi64x(i as i64)
}

#[inline(always)]
pub(crate) unsafe fn veq64_vo_vm_vm(x: VMask, y: VMask) -> Vopmask {
    _mm256_cmpeq_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vadd64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm256_add_epi64(x, y)
}


#[inline(always)]
pub(crate) unsafe fn vadd_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_add_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_sub_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmul_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_mul_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vdiv_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_div_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vrec_vd_vd(x: VDouble) -> VDouble {
    _mm256_div_pd(_mm256_set1_pd(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vd_vd(x: VDouble) -> VDouble {
    _mm256_sqrt_pd(x)
}

#[inline(always)]
pub(crate) unsafe fn vabs_vd_vd(d: VDouble) -> VDouble {
    _mm256_andnot_pd(_mm256_set1_pd(-0.0), d)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vd_vd(d: VDouble) -> VDouble {
    _mm256_xor_pd(_mm256_set1_pd(-0.0), d)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fmadd_pd(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_add_pd(_mm256_mul_pd(x, y), z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fmsub_pd(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_sub_pd(_mm256_mul_pd(x, y), z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmlanp_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fnmadd_pd(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vmax_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_max_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmin_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_min_pd(x, y)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapp_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fmsub_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmanp_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fnmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmann_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm256_fnmsub_pd(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_EQ_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_NEQ_UQ))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_LT_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_LE_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_GT_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_GE_OQ))
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
pub(crate) unsafe fn vxor_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm_xor_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi_vo_vi(m: Vopmask, y: VInt) -> VInt {
    _mm_andnot_si128(_mm256_castsi256_si128(m), y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vo_vi(m: Vopmask, y: VInt) -> VInt {
    _mm_and_si128(_mm256_castsi256_si128(m), y)
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
pub(crate) unsafe fn veq_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    _mm256_castsi128_si256(_mm_cmpeq_epi32(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    _mm256_castsi128_si256(_mm_cmpgt_epi32(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi_vo_vi_vi(m: Vopmask, x: VInt, y: VInt) -> VInt {
    _mm_blendv_epi8(y, x, _mm256_castsi256_si128(m))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vd_vd(o: Vopmask, x: VDouble, y: VDouble) -> VDouble {
    _mm256_blendv_pd(y, x, _mm256_castsi256_pd(o))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_d_d(o: Vopmask, v1: f64, v0: f64) -> VDouble {
    _mm256_permutevar_pd(_mm256_set_pd(v1, v0, v1, v0), o)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vo_vo_d_d_d_d(
    o0: Vopmask,
    o1: Vopmask,
    o2: Vopmask,
    d0: f64,
    d1: f64,
    d2: f64,
    d3: f64
) -> VDouble {
    let v = _mm256_castpd_si256(
        vsel_vd_vo_vd_vd(
            o0,
            _mm256_castsi256_pd(_mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0)),
            vsel_vd_vo_vd_vd(
                o1,
                _mm256_castsi256_pd(_mm256_set_epi32(3, 2, 3, 2, 3, 2, 3, 2)),
                vsel_vd_vo_vd_vd(
                    o2,
                    _mm256_castsi256_pd(_mm256_set_epi32(5, 4, 5, 4, 5, 4, 5, 4)),
                    _mm256_castsi256_pd(_mm256_set_epi32(7, 6, 7, 6, 7, 6, 7, 6))
                )
            )
        )
    );

    _mm256_castsi256_pd(
        _mm256_permutevar8x32_epi32(_mm256_castpd_si256(_mm256_set_pd(d3, d2, d1, d0)), v)
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vo_d_d_d(
    o0: Vopmask,
    o1: Vopmask,
    d0: f64,
    d1: f64,
    d2: f64
) -> VDouble {
    vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o1, d0, d1, d2, d2)
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vd(d: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(vabs_vd_vd(d), _mm256_set1_pd(f64::INFINITY), _CMP_EQ_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vd(d: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(d, _mm256_set1_pd(f64::INFINITY), _CMP_EQ_OQ))
}

#[inline(always)]
pub(crate) unsafe fn visminf_vo_vd(d: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(d, _mm256_set1_pd(f64::NEG_INFINITY), _CMP_EQ_OQ))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vd(d: VDouble) -> Vopmask {
    vreinterpret_vm_vd(_mm256_cmp_pd(d, d, _CMP_NEQ_UQ))
}

#[inline(always)]
pub(crate) unsafe fn vcast_d_vd(v: VDouble) -> f64 {
    let mut s: [f64; 4] = [0.0; 4];
    _mm256_storeu_pd(s.as_mut_ptr(), v);
    s[0]
}

#[inline(always)]
pub(crate) unsafe fn vload_vd_p(ptr: *const f64) -> VDouble {
    _mm256_load_pd(ptr)
}

#[inline(always)]
pub(crate) unsafe fn vloadu_vd_p(ptr: *const f64) -> VDouble {
    _mm256_loadu_pd(ptr)
}

#[inline(always)]
pub(crate) unsafe fn vstore_v_p_vd(ptr: *mut f64, v: VDouble) {
    _mm256_store_pd(ptr, v)
}

#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vd(ptr: *mut f64, v: VDouble) {
    _mm256_storeu_pd(ptr, v)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vd_p_vi(ptr: *const f64, vi: VInt) -> VDouble {
    _mm256_i32gather_pd(ptr, vi, 8)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_vm(vm: VMask) -> VInt2 {
    vm
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_vi2(vi: VInt2) -> VMask {
    vi
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi2_vf(vf: VFloat) -> VInt2 {
    vcast_vi2_vm(_mm256_cvtps_epi32(vf))
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi2_vf(vf: VFloat) -> VInt2 {
    vcast_vi2_vm(_mm256_cvttps_epi32(vf))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vi2(vi: VInt2) -> VFloat {
    _mm256_cvtepi32_ps(vcast_vm_vi2(vi))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_f(f: f32) -> VFloat {
    _mm256_set1_ps(f)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_i(i: i32) -> VInt2 {
    _mm256_set1_epi32(i)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vf(vf: VFloat) -> VMask {
    _mm256_castps_si256(vf)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vm(vm: VMask) -> VFloat {
    _mm256_castsi256_ps(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vi2(vi: VInt2) -> VFloat {
    vreinterpret_vf_vm(vcast_vm_vi2(vi))
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vi2_vf(vf: VFloat) -> VInt2 {
    vcast_vi2_vm(vreinterpret_vm_vf(vf))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_add_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_sub_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmul_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_mul_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vdiv_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_div_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vrec_vf_vf(x: VFloat) -> VFloat {
    vdiv_vf_vf_vf(vcast_vf_f(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vf_vf(x: VFloat) -> VFloat {
    _mm256_sqrt_ps(x)
}

#[inline(always)]
pub(crate) unsafe fn vabs_vf_vf(f: VFloat) -> VFloat {
    vreinterpret_vf_vm(
        vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(f))
    )
}

#[inline(always)]
pub(crate) unsafe fn vneg_vf_vf(d: VFloat) -> VFloat {
    vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(d)))
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fmadd_ps(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_add_ps(_mm256_mul_ps(x, y), z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmlapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fmsub_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fnmadd_ps(x, y, z)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_sub_ps(z, _mm256_mul_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vmax_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_max_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmin_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_min_ps(x, y)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fmsub_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fnmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmann_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm256_fnmsub_ps(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_EQ_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_NEQ_UQ))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LT_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LE_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_GT_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_GE_OQ))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_add_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_sub_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi2_vi2(e: VInt2) -> VInt2 {
    vsub_vi2_vi2_vi2(vcast_vi2_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_and_si256(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_andnot_si256(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_or_si256(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_xor_si256(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi2_vo_vi2(x: Vopmask, y: VInt2) -> VInt2 {
    vand_vi2_vi2_vi2(vcast_vi2_vm(x), y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vo_vi2(x: Vopmask, y: VInt2) -> VInt2 {
    vandnot_vi2_vi2_vi2(vcast_vi2_vm(x), y)
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
    _mm256_slli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
    _mm256_srli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
    _mm256_srai_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    _mm256_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    _mm256_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn veq_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    _mm256_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi2_vo_vi2_vi2(m: Vopmask, x: VInt2, y: VInt2) -> VInt2 {
    _mm256_blendv_epi8(y, x, m)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vf_vf(o: Vopmask, x: VFloat, y: VFloat) -> VFloat {
    _mm256_blendv_ps(y, x, _mm256_castsi256_ps(o))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_f_f(o: Vopmask, v1: f32, v0: f32) -> VFloat {
    vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vo_f_f_f(
    o0: Vopmask,
    o1: Vopmask,
    d0: f32,
    d1: f32,
    d2: f32
) -> VFloat {
    vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vo_vo_f_f_f_f(
    o0: Vopmask,
    o1: Vopmask,
    o2: Vopmask,
    d0: f32,
    d1: f32,
    d2: f32,
    d3: f32
) -> VFloat {
    vsel_vf_vo_vf_vf(
        o0,
        vcast_vf_f(d0),
        vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3))
    )
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
pub(crate) unsafe fn visminf_vo_vf(d: VFloat) -> Vopmask {
    veq_vo_vf_vf(d, vcast_vf_f(-f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vf(d: VFloat) -> Vopmask {
    vneq_vo_vf_vf(d, d)
}

#[inline(always)]
pub(crate) unsafe fn vcast_f_vf(v: VFloat) -> f32 {
    let mut s: [f32; 8] = [0.0; 8];
    _mm256_storeu_ps(s.as_mut_ptr(), v);
    s[0]
}

#[inline(always)]
pub(crate) unsafe fn vload_vf_p(ptr: *const f32) -> VFloat {
    _mm256_load_ps(ptr)
}

#[inline(always)]
pub(crate) unsafe fn vloadu_vf_p(ptr: *const f32) -> VFloat {
    _mm256_loadu_ps(ptr)
}

#[inline(always)]
pub(crate) unsafe fn vstore_v_p_vf(ptr: *mut f32, v: VFloat) {
    _mm256_store_ps(ptr, v)
}

#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vf(ptr: *mut f32, v: VFloat) {
    _mm256_storeu_ps(ptr, v)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vf_p_vi2(ptr: *const f32, vi2: VInt2) -> VFloat {
    _mm256_i32gather_ps(ptr, vi2, 4)
}

const PNMASK: __m256d = unsafe { std::mem::transmute([0.0f64, -0.0f64, 0.0f64, -0.0f64]) };
const NPMASK: __m256d = unsafe { std::mem::transmute([-0.0f64, 0.0f64, -0.0f64, 0.0f64]) };
const PNMASKF: __m256 = unsafe {
    std::mem::transmute([0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32])
};
const NPMASKF: __m256 = unsafe {
    std::mem::transmute([-0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32])
};

#[inline(always)]
pub(crate) unsafe fn vposneg_vd_vd(d: VDouble) -> VDouble {
    vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(PNMASK)))
}

#[inline(always)]
pub(crate) unsafe fn vnegpos_vd_vd(d: VDouble) -> VDouble {
    vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(NPMASK)))
}

#[inline(always)]
pub(crate) unsafe fn vposneg_vf_vf(d: VFloat) -> VFloat {
    vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(PNMASKF)))
}

#[inline(always)]
pub(crate) unsafe fn vnegpos_vf_vf(d: VFloat) -> VFloat {
    vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(NPMASKF)))
}

#[inline(always)]
pub(crate) unsafe fn vsubadd_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm256_addsub_pd(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsubadd_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm256_addsub_ps(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmlsubadd_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    vmla_vd_vd_vd_vd(x, y, vnegpos_vd_vd(z))
}

#[inline(always)]
pub(crate) unsafe fn vmlsubadd_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x, y, vnegpos_vf_vf(z))
}

#[inline(always)]
pub(crate) unsafe fn vrev21_vd_vd(d0: VDouble) -> VDouble {
    _mm256_shuffle_pd(d0, d0, 0b0101)
}

#[inline(always)]
pub(crate) unsafe fn vreva2_vd_vd(d0: VDouble) -> VDouble {
    let d = _mm256_permute2f128_pd(d0, d0, 1);
    _mm256_shuffle_pd(d, d, 0b1010)
}

#[inline(always)]
pub(crate) unsafe fn vstream_v_p_vd(ptr: *mut f64, v: VDouble) {
    _mm256_stream_pd(ptr, v)
}

#[inline(always)]
pub(crate) unsafe fn vscatter2_v_p_i_i_vd(ptr: *mut f64, offset: i32, step: i32, v: VDouble) {
    _mm_store_pd(ptr.add(((offset + step * 0) as usize) * 2).cast(), _mm256_extractf128_pd(v, 0));
    _mm_store_pd(ptr.add(((offset + step * 1) as usize) * 2).cast(), _mm256_extractf128_pd(v, 1));
}

#[inline(always)]
pub(crate) unsafe fn vsscatter2_v_p_i_i_vd(ptr: *mut f64, offset: i32, step: i32, v: VDouble) {
    _mm_stream_pd(ptr.add(((offset + step * 0) as usize) * 2).cast(), _mm256_extractf128_pd(v, 0));
    _mm_stream_pd(ptr.add(((offset + step * 1) as usize) * 2).cast(), _mm256_extractf128_pd(v, 1));
}

#[inline(always)]
pub(crate) unsafe fn vrev21_vf_vf(d0: VFloat) -> VFloat {
    _mm256_shuffle_ps(d0, d0, 0b10_11_00_01)
}

#[inline(always)]
pub(crate) unsafe fn vreva2_vf_vf(d0: VFloat) -> VFloat {
    let d = _mm256_permute2f128_ps(d0, d0, 1);
    _mm256_shuffle_ps(d, d, 0b01_00_11_10)
}

#[inline(always)]
pub(crate) unsafe fn vstream_v_p_vf(ptr: *mut f32, v: VFloat) {
    _mm256_stream_ps(ptr, v)
}

#[inline(always)]
pub(crate) unsafe fn vscatter2_v_p_i_i_vf(ptr: *mut f32, offset: i32, step: i32, v: VFloat) {
    _mm_storel_pd(
        ptr.add(((offset + step * 0) as usize) * 2).cast(),
        _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 0)))
    );
    _mm_storeh_pd(
        ptr.add(((offset + step * 1) as usize) * 2).cast(),
        _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 0)))
    );
    _mm_storel_pd(
        ptr.add(((offset + step * 2) as usize) * 2).cast(),
        _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 1)))
    );
    _mm_storeh_pd(
        ptr.add(((offset + step * 3) as usize) * 2).cast(),
        _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 1)))
    );
}

#[inline(always)]
pub(crate) unsafe fn vsscatter2_v_p_i_i_vf(ptr: *mut f32, offset: i32, step: i32, v: VFloat) {
    vscatter2_v_p_i_i_vf(ptr, offset, step, v)
}

#[inline(always)]
pub(crate) unsafe fn loadu_vq_p(p: *const std::ffi::c_void) -> Vquad {
    let mut vq = [0u8; (1 << 2) * 16];
    std::ptr::copy_nonoverlapping(p.cast(), vq.as_mut_ptr(), (1 << 2) * 16);
    transmute(vq)
}

#[inline(always)]
pub(crate) unsafe fn cast_vq_aq(aq: Vargquad) -> Vquad {
    let mut vq = [0u8; (1 << 2) * 16];
    std::ptr::copy_nonoverlapping(aq.as_ptr(), vq.as_mut_ptr(), (1 << 2) * 16);
    transmute(vq)
}

#[inline(always)]
pub(crate) unsafe fn cast_aq_vq(vq: Vquad) -> Vargquad {
    let mut aq = [0u8; (1 << 2) * 16];
    std::ptr::copy_nonoverlapping(vq.as_ptr(), aq.as_mut_ptr(), (1 << 2) * 16);
    transmute(aq)
}

#[inline(always)]
pub(crate) unsafe fn vtestallzeros_i_vo64(g: Vopmask) -> i32 {
    (_mm_movemask_epi8(
        _mm_or_si128(_mm256_extractf128_si256::<0>(g), _mm256_extractf128_si256::<1>(g))
    ) == 0) as i32
}

#[inline(always)]
pub(crate) unsafe fn vsel_vm_vo64_vm_vm(o: Vopmask, x: VMask, y: VMask) -> VMask {
    _mm256_blendv_epi8(y, x, o)
}

#[inline(always)]
pub(crate) unsafe fn vsub64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm256_sub_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg64_vm_vm(x: VMask) -> VMask {
    _mm256_sub_epi64(vcast_vm_i_i(0, 0), x)
}

#[inline(always)]
pub(crate) unsafe fn vgt64_vo_vm_vm(x: VMask, y: VMask) -> Vopmask {
    _mm256_cmpgt_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsll64_vm_vm_i<const C: i32>(x: VMask) -> VMask {
    _mm256_slli_epi64::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl64_vm_vm_i<const C: i32>(x: VMask) -> VMask {
    _mm256_srli_epi64::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_vi(vi: VInt) -> VMask {
    _mm256_cvtepi32_epi64(vi)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi_vm(vm: VMask) -> VInt {
    _mm_or_si128(
        _mm_castps_si128(
            _mm_shuffle_ps(_mm_castsi128_ps(_mm256_castsi256_si128(vm)), _mm_set1_ps(0.0), 0x08)
        ),
        _mm_castps_si128(
            _mm_shuffle_ps(
                _mm_set1_ps(0.0),
                _mm_castsi128_ps(_mm256_extractf128_si256(vm, 1)),
                0x80
            )
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vi64(v: VInt64) -> VMask {
    v
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vi64_vm(m: VMask) -> VInt64 {
    m
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vu64(v: VUInt64) -> VMask {
    v
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vu64_vm(m: VMask) -> VUInt64 {
    m
}
