use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vprefetch_v_p(_ptr: *const std::ffi::c_void) {
    // Empty implementation
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo32(g: Vopmask) -> i32 {
    let x0 = vand_u32(vget_low_u32(g), vget_high_u32(g));
    let x1 = vpmin_u32(x0, x0);
    vget_lane_u32(x1, 0) as i32
}

#[inline(always)]
pub(crate) unsafe fn vloaduf(p: *const f32) -> VFloat {
    vld1q_f32(p)
}

#[inline(always)]
pub(crate) unsafe fn vstoreuf(p: *mut f32, v: VFloat) {
    vst1q_f32(p, v)
}

#[inline(always)]
pub(crate) unsafe fn vloadu_vi2_p(p: *const i32) -> VInt2 {
    vld1q_s32(p)
}

#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vi2(p: *mut i32, v: VInt2) {
    vst1q_s32(p, v)
}

// Bitwise operations
#[inline(always)]
pub(crate) unsafe fn vand_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vandq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vbicq_u32(y, x) // Note: arguments are swapped compared to x86
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vorrq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    veorq_u32(x, y)
}

// Mask operations
#[inline(always)]
pub(crate) unsafe fn vand_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vandq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vbicq_u32(y, x) // Note: arguments are swapped compared to x86
}

#[inline(always)]
pub(crate) unsafe fn vor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    vorrq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    veorq_u32(x, y)
}

// 64-bit mask operations
#[inline(always)]
pub(crate) unsafe fn vand_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vandq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vbicq_u32(y, x) // Note: arguments swapped compared to x86
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    vorrq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    veorq_u32(x, y)
}

// 32-bit mask operations
#[inline(always)]
pub(crate) unsafe fn vand_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vandq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vbicq_u32(y, x) // Note: arguments swapped compared to x86
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    vorrq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    veorq_u32(x, y)
}

// Cast operations
#[inline(always)]
pub(crate) unsafe fn vcast_vo32_vo64(m: Vopmask) -> Vopmask {
    vuzpq_u32(m, m).0
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo64_vo32(m: Vopmask) -> Vopmask {
    vzipq_u32(m, m).0
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_i_i(i0: i32, i1: i32) -> VMask {
    let val = (i0 as u32) | ((i1 as u32) << 32);
    vdupq_n_u32(val)
}

#[inline(always)]
pub(crate) unsafe fn veq64_vo_vm_vm(x: VMask, y: VMask) -> Vopmask {
    let t = vceqq_u32(x, y);
    vandq_u32(t, vrev64q_u32(t))
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi2_vf(d: VFloat) -> VInt2 {
    let a: uint32x4_t = std::mem::transmute(vdupq_n_f32(0.5f32));
    let b: uint32x4_t = std::mem::transmute(vdupq_n_f32(-0.0f32));
    let d_casted = std::mem::transmute(d);
    let x = std::mem::transmute(vorrq_u32(vandq_u32(d_casted, b), a));
    vcvtq_s32_f32(vaddq_f32(d, x))
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi2_vf(vf: VFloat) -> VInt2 {
    vcvtq_s32_f32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vi2(vi: VInt2) -> VFloat {
    vcvtq_f32_s32(vi)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vf_vf(vd: VFloat) -> VFloat {
    vcast_vf_vi2(vtruncate_vi2_vf(vd))
}

#[inline(always)]
pub(crate) unsafe fn vrint_vf_vf(vd: VFloat) -> VFloat {
    vcast_vf_vi2(vrint_vi2_vf(vd))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_f(f: f32) -> VFloat {
    vdupq_n_f32(f)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_i(i: i32) -> VInt2 {
    vdupq_n_s32(i)
}

// Reinterpret casts
#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vf(vf: VFloat) -> VMask {
    vreinterpretq_u32_f32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vm(vm: VMask) -> VFloat {
    vreinterpretq_f32_u32(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vi2(vm: VInt2) -> VFloat {
    vreinterpretq_f32_s32(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vi2_vf(vf: VFloat) -> VInt2 {
    vreinterpretq_s32_f32(vf)
}

// Basic arithmetic operations
#[inline(always)]
pub(crate) unsafe fn vadd_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vaddq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vsubq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmul_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vmulq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vabs_vf_vf(f: VFloat) -> VFloat {
    vabsq_f32(f)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vf_vf(f: VFloat) -> VFloat {
    vnegq_f32(f)
}

#[cfg(target_feature = "fma")]
pub(crate) mod fma_ops {
    use super::*;
    #[inline(always)]
    pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vfmaq_f32(z, x, y)
    }

    #[inline(always)]
    pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vfmsq_f32(z, x, y)
    }

    #[inline(always)]
    pub(crate) unsafe fn vfma_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vfmaq_f32(z, x, y)
    }

    #[inline(always)]
    pub(crate) unsafe fn vfmanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vfmsq_f32(z, x, y)
    }

    #[inline(always)]
    pub(crate) unsafe fn vfmapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vneg_vf_vf(vfmanp_vf_vf_vf_vf(x, y, z))
    }

    #[inline(always)]
    pub(crate) unsafe fn vmlapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vneg_vf_vf(vfmanp_vf_vf_vf_vf(x, y, z))
    }

    #[inline(always)]
    pub(crate) unsafe fn vdiv_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
        // Initial estimate
        let mut t = vrecpeq_f32(y);

        // First refinement
        t = vmulq_f32(t, vrecpsq_f32(y, t));

        // Second refinement with FMA
        let one = vdupq_n_f32(1.0);
        t = vfmaq_f32(t, vfmsq_f32(one, y, t), t);

        // Compute quotient
        let u = vmulq_f32(x, t);

        // Final refinement
        vfmaq_f32(u, vfmsq_f32(x, y, u), t)
    }

    #[inline(always)]
    pub(crate) unsafe fn vsqrt_vf_vf(d: VFloat) -> VFloat {
        // Initial estimate for reciprocal square root
        let mut x = vrsqrteq_f32(d);

        // Two Newton-Raphson iterations
        x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));
        x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));

        // Compute square root
        let mut u = vmulq_f32(x, d);

        // One iteration of refinement
        let half = vdupq_n_f32(0.5);
        u = vfmaq_f32(u, vfmsq_f32(d, u, u), vmulq_f32(x, half));

        // Handle zero input
        let zero = vdupq_n_f32(0.0);
        vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(u), vceqq_f32(d, zero)))
    }

    #[inline(always)]
    pub(crate) unsafe fn vrec_vf_vf(y: VFloat) -> VFloat {
        // Initial estimate
        let mut t = vrecpeq_f32(y);

        // First refinement
        t = vmulq_f32(t, vrecpsq_f32(y, t));

        // Second refinement with FMA
        let one = vdupq_n_f32(1.0);
        t = vfmaq_f32(t, vfmsq_f32(one, y, t), t);

        // Final refinement
        vfmaq_f32(t, vfmsq_f32(one, y, t), t)
    }

    #[inline(always)]
    pub(crate) unsafe fn vrecsqrt_vf_vf(d: VFloat) -> VFloat {
        // Initial estimate
        let mut x = vrsqrteq_f32(d);

        // First refinement
        x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));

        // Final refinement with FMA
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        vfmaq_f32(x, vfmsq_f32(one, x, vmulq_f32(x, d)), vmulq_f32(x, half))
    }
}

#[cfg(not(target_feature = "fma"))]
pub(crate) mod fma_ops {
    use super::*;

    #[inline(always)]
    pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vmlaq_f32(z, x, y)
    }

    #[inline(always)]
    pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vmlsq_f32(z, x, y)
    }

    #[inline(always)]
    pub(crate) unsafe fn vmlapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
        vneg_vf_vf(vmlsq_f32(z, x, y))
    }

    #[inline(always)]
    pub(crate) unsafe fn vdiv_vf_vf_vf(n: VFloat, d: VFloat) -> VFloat {
        let mut x: float32x4_t = vrecpeq_f32(d);
        x = vmulq_f32(x, vrecpsq_f32(d, x));
        let t: float32x4_t = vmulq_f32(n, x);
        vmlsq_f32(vaddq_f32(t, t), vmulq_f32(t, x), d)
    }

    #[inline(always)]
    pub(crate) unsafe fn vsqrt_vf_vf(d: VFloat) -> VFloat {
        let mut x: float32x4_t = vrsqrteq_f32(d);
        x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));
        let mut u: float32x4_t = vmulq_f32(x, d);
        u = vmlaq_f32(u, vmlsq_f32(d, u, u), vmulq_f32(x, vdupq_n_f32(0.5)));
        vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(u), vceqq_f32(d, vdupq_n_f32(0.0))))
    }

    #[inline(always)]
    pub(crate) unsafe fn vrec_vf_vf(d: VFloat) -> VFloat {
        let mut x: float32x4_t = vrecpeq_f32(d);
        x = vmulq_f32(x, vrecpsq_f32(d, x));
        vmlsq_f32(vaddq_f32(x, x), vmulq_f32(x, x), d)
    }

    #[inline(always)]
    pub(crate) unsafe fn vrecsqrt_vf_vf(d: VFloat) -> VFloat {
        let mut x: float32x4_t = vrsqrteq_f32(d);
        x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));
        vmlaq_f32(
            x,
            vmlsq_f32(vdupq_n_f32(1.0), x, vmulq_f32(x, d)),
            vmulq_f32(x, vdupq_n_f32(0.5))
        )
    }
}

pub(crate) use fma_ops::*;

use crate::sleef_types::{ VDouble, VFloat, VInt, VInt2, VMask, Vopmask };

// Float comparisons and operations
#[inline(always)]
pub(crate) unsafe fn vmax_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vmaxq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmin_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vminq_f32(x, y)
}

// Float comparison masks
#[inline(always)]
pub(crate) unsafe fn veq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vceqq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vmvnq_u32(vceqq_f32(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vcltq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vcleq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vcgtq_f32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    vcgeq_f32(x, y)
}

// Integer vector operations
#[inline(always)]
pub(crate) unsafe fn vadd_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    vaddq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    vsubq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi2_vi2(e: VInt2) -> VInt2 {
    vnegq_s32(e)
}

// Integer bitwise operations
#[inline(always)]
pub(crate) unsafe fn vand_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    vandq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    vbicq_s32(y, x)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    vorrq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    veorq_s32(x, y)
}

// Mask and integer operations
#[inline(always)]
pub(crate) unsafe fn vand_vi2_vo_vi2(x: Vopmask, y: VInt2) -> VInt2 {
    vreinterpretq_s32_u32(vandq_u32(x, vreinterpretq_u32_s32(y)))
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vo_vi2(x: Vopmask, y: VInt2) -> VInt2 {
    vreinterpretq_s32_u32(vbicq_u32(vreinterpretq_u32_s32(y), x))
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    vceqq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vf_p_vi2(ptr: *const f32, vi2: VInt2) -> VFloat {
    std::mem::transmute([
        *ptr.offset(vgetq_lane_s32(vi2, 0) as isize),
        *ptr.offset(vgetq_lane_s32(vi2, 1) as isize),
        *ptr.offset(vgetq_lane_s32(vi2, 2) as isize),
        *ptr.offset(vgetq_lane_s32(vi2, 3) as isize),
    ])
}

#[inline(always)]
pub(crate) unsafe fn vgt_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    vreinterpretq_s32_u32(vcgtq_s32(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    vcgtq_s32(x, y)
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
    veq_vo_vf_vf(d, vcast_vf_f(f32::NEG_INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vf(d: VFloat) -> Vopmask {
    vneq_vo_vf_vf(d, d)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_f_f(o: Vopmask, v1: f32, v0: f32) -> VFloat {
    vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vf_vf(mask: Vopmask, x: VFloat, y: VFloat) -> VFloat {
    vreinterpretq_f32_u32(vbslq_u32(mask, vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi2_vo_vi2_vi2(m: Vopmask, x: VInt2, y: VInt2) -> VInt2 {
    vreinterpretq_s32_u32(vbslq_u32(m, vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)))
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi2_vi2_i<const N: i32>(x: VInt2) -> VInt2 {
    vshlq_n_s32::<N>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi2_vi2_i<const N: i32>(x: VInt2) -> VInt2 {
    vshrq_n_s32::<N>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi2_vi2_i<const N: i32>(x: VInt2) -> VInt2 {
    vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(x)))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    vaddq_f64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_d(x: f64) -> VDouble {
    vdupq_n_f64(x)
}

#[inline(always)]
pub(crate) unsafe fn vmul_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    vmulq_f64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vd_vd(x: VDouble) -> VDouble {
    vnegq_f64(x)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vd_vd(mask: Vopmask, x: VDouble, y: VDouble) -> VDouble {
    vreinterpretq_f64_u64(
        vbslq_u64(
            vreinterpretq_u64_u32(vcast_vo32_vo64(mask)),
            vreinterpretq_u64_f64(x),
            vreinterpretq_u64_f64(y)
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vd_vd(x: VDouble) -> VDouble {
    vsqrtq_f64(x)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    vsubq_f64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vd_vm(x: VMask) -> VDouble {
    vreinterpretq_f64_u32(x)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vd(x: VDouble) -> VMask {
    vreinterpretq_u32_f64(x)
}

#[inline(always)]
pub(crate) unsafe fn vrec_vd_vd(x: VDouble) -> VDouble {
    vrecpeq_f64(x)
}

#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    vmlaq_f64(z, x, y)
}

#[inline(always)]
pub(crate) unsafe fn vabs_vd_vd(x: VDouble) -> VDouble {
    vabsq_f64(x)
}

#[inline(always)]
pub(crate) unsafe fn vadd64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vaddq_u32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    vaddq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_vi(vi: VInt) -> VDouble {
    vcvtq_f64_s64(vmovl_s32(vget_low_s32(vi)))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi_i(i: i32) -> VInt {
    let zero = vdupq_n_s32(0);
    vsetq_lane_s32(i, vsetq_lane_s32(i, zero, 0), 1)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi_vi(e: VInt) -> VInt {
    vsub_vi_vi_vi(vcast_vi_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vm_vi(x: VInt) -> VMask {
    vreinterpretq_u32_s32(x)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpretq_u32_u64(vceqq_f64(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpretq_u32_u64(vcgeq_f64(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpretq_u32_u64(vcgtq_f64(x, y))
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vd(d: VDouble) -> Vopmask {
    veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(f64::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpretq_u32_u64(vcleq_f64(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vreinterpretq_u32_u64(vcgtq_f64(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(x, y)))
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi_vi_i<const N: i32>(x: VInt) -> VInt {
    vshlq_n_s32::<N>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi_vi_i<const N: i32>(x: VInt) -> VInt {
    vshrq_n_s32::<N>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    vsubq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi_vd(vd: VDouble) -> VInt {
    let truncated = vcvtq_s64_f64(vd);
    vreinterpretq_s32_s64(truncated)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    vandq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vi_vm(vi: VMask) -> VInt {
    vreinterpretq_s32_u32(vi)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi_vo_vi_vi(m: Vopmask, x: VInt, y: VInt) -> VInt {
    vreinterpretq_s32_u32(vbslq_u32(m, vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)))
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi_vi_i<const N: i32>(x: VInt) -> VInt {
    vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(x)))
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vo_vi(m: Vopmask, y: VInt) -> VInt {
    vandq_s32(vreinterpretq_s32_u32(m), y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    vbicq_s32(y, x)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    vceqq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vd_p_vi(ptr: *const f64, vi: VInt) -> VDouble {
    std::mem::transmute([
        *ptr.offset(vgetq_lane_s32(vi, 0) as isize),
        *ptr.offset(vgetq_lane_s32(vi, 1) as isize),
    ])
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    vcgtq_s32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vd(d: VDouble) -> Vopmask {
    vreinterpretq_u32_u64(vceqzq_u64(vceqq_f64(d, d)))
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vd(d: VDouble) -> Vopmask {
    let inf = vdupq_n_f64(f64::INFINITY);
    vreinterpretq_u32_u64(vceqq_f64(d, inf))
}

#[inline(always)]
pub(crate) unsafe fn vmax_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    vmaxq_f64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmin_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    vminq_f64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    vsubq_f64(vmulq_f64(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vd_vd(vd: VDouble) -> VDouble {
    vrndnq_f64(vd)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi_vd(vd: VDouble) -> VInt {
    let lo = vcvtnq_s64_f64(vd);
    vreinterpretq_s32_s64(lo)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_d_d(o: Vopmask, v1: f64, v0: f64) -> VDouble {
    vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0))
}

#[inline(always)]
pub(crate) unsafe fn vsrl64_vm_vm_i<const C: i32>(x: VMask) -> VMask {
    vreinterpretq_u32_u64(vshrq_n_u64::<C>(vreinterpretq_u64_u32(x)))
}

#[inline(always)]
pub(crate) unsafe fn vsub64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    vreinterpretq_u32_u64(vsubq_u64(vreinterpretq_u64_u32(x), vreinterpretq_u64_u32(y)))
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo64(g: VMask) -> i32 {
    (vminvq_u32(g) == 0xffffffff_u32) as i32
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vd_vd(vd: VDouble) -> VDouble {
    vrndq_f64(vd)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vm(vm: VMask) -> VFloat {
    vreinterpretq_f32_u32(vm)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_vf(vf: VFloat) -> VMask {
    vreinterpretq_u32_f32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vo(vf: Vopmask) -> VFloat {
    vcast_vf_vm(vf)
}
