#![allow(unused)]

use crate::{
    arch_simd::sleef::arch::helper::{
        vabs_vf_vf,
        vadd_vf_vf_vf,
        vand_vi2_vi2_vi2,
        vand_vm_vm_vm,
        vcast_vf_f,
        vcast_vi2_i,
        vfma_vf_vf_vf_vf,
        vfmanp_vf_vf_vf_vf,
        vfmapn_vf_vf_vf_vf,
        vmla_vf_vf_vf_vf,
        vmul_vf_vf_vf,
        vneg_vf_vf,
        vrec_vf_vf,
        vreinterpret_vf_vi2,
        vreinterpret_vf_vm,
        vreinterpret_vi2_vf,
        vreinterpret_vm_vf,
        vsel_vf_vo_f_f,
        vsel_vf_vo_vf_vf,
        vsqrt_vf_vf,
        vsub_vf_vf_vf,
        vxor_vm_vm_vm,
    },
    VFloat,
    Vopmask,
};

// #if !(defined(ENABLE_SVE) || defined(ENABLE_SVENOFMA) || defined(ENABLE_RVVM1) || defined(ENABLE_RVVM1NOFMA) || defined(ENABLE_RVVM2) || defined(ENABLE_RVVM2NOFMA))
// #if !defined(SLEEF_ENABLE_CUDA)
#[derive(Clone, Copy)]
pub(crate) struct VFloat2 {
    pub(crate) x: VFloat,
    pub(crate) y: VFloat,
}

pub(crate) fn vf2getx_vf_vf2(v: VFloat2) -> VFloat {
    v.x
}
pub(crate) fn vf2gety_vf_vf2(v: VFloat2) -> VFloat {
    v.y
}
pub(crate) fn vf2setxy_vf2_vf_vf(x: VFloat, y: VFloat) -> VFloat2 {
    VFloat2 { x, y }
}
pub(crate) fn vf2setx_vf2_vf2_vf(mut v: VFloat2, x: VFloat) -> VFloat2 {
    v.x = x;
    v
}
pub(crate) fn vf2sety_vf2_vf2_vf(mut v: VFloat2, y: VFloat) -> VFloat2 {
    v.y = y;
    v
}

// #endif

#[inline(always)]
pub(crate) unsafe fn vupper_vf_vf(d: VFloat) -> VFloat {
    vreinterpret_vf_vi2(vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0xfffff000u32 as i32)))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf2_vf_vf(h: VFloat, l: VFloat) -> VFloat2 {
    vf2setxy_vf2_vf_vf(h, l)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf2_f_f(h: f32, l: f32) -> VFloat2 {
    vf2setxy_vf2_vf_vf(vcast_vf_f(h), vcast_vf_f(l))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf2_d(d: f64) -> VFloat2 {
    vf2setxy_vf2_vf_vf(vcast_vf_f(d as f32), vcast_vf_f((d - (d as f32 as f64)) as f32))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf2_vo_vf2_vf2(m: Vopmask, x: VFloat2, y: VFloat2) -> VFloat2 {
    vf2setxy_vf2_vf_vf(
        vsel_vf_vo_vf_vf(m, vf2getx_vf_vf2(x), vf2getx_vf_vf2(y)),
        vsel_vf_vo_vf_vf(m, vf2gety_vf_vf2(x), vf2gety_vf_vf2(y))
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf2_vo_f_f_f_f(
    o: Vopmask,
    x1: f32,
    y1: f32,
    x0: f32,
    y0: f32
) -> VFloat2 {
    vf2setxy_vf2_vf_vf(vsel_vf_vo_f_f(o, x1, x0), vsel_vf_vo_f_f(o, y1, y0))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf2_vo_vo_d_d_d(
    o0: Vopmask,
    o1: Vopmask,
    d0: f64,
    d1: f64,
    d2: f64
) -> VFloat2 {
    vsel_vf2_vo_vf2_vf2(
        o0,
        vcast_vf2_d(d0),
        vsel_vf2_vo_vf2_vf2(o1, vcast_vf2_d(d1), vcast_vf2_d(d2))
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf2_vo_vo_vo_d_d_d_d(
    o0: Vopmask,
    o1: Vopmask,
    o2: Vopmask,
    d0: f64,
    d1: f64,
    d2: f64,
    d3: f64
) -> VFloat2 {
    vsel_vf2_vo_vf2_vf2(
        o0,
        vcast_vf2_d(d0),
        vsel_vf2_vo_vf2_vf2(
            o1,
            vcast_vf2_d(d1),
            vsel_vf2_vo_vf2_vf2(o2, vcast_vf2_d(d2), vcast_vf2_d(d3))
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn vabs_vf2_vf2(x: VFloat2) -> VFloat2 {
    let neg_zero = vreinterpret_vm_vf(vcast_vf_f(-0.0));
    let x_high = vreinterpret_vm_vf(vf2getx_vf_vf2(x));
    let x_low = vreinterpret_vm_vf(vf2gety_vf_vf2(x));
    let mask = vand_vm_vm_vm(neg_zero, x_high);

    vcast_vf2_vf_vf(
        vreinterpret_vf_vm(vxor_vm_vm_vm(mask, x_high)),
        vreinterpret_vf_vm(vxor_vm_vm_vm(mask, x_low))
    )
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_3vf(v0: VFloat, v1: VFloat, v2: VFloat) -> VFloat {
    vadd_vf_vf_vf(vadd_vf_vf_vf(v0, v1), v2)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_4vf(v0: VFloat, v1: VFloat, v2: VFloat, v3: VFloat) -> VFloat {
    vadd_vf_3vf(vadd_vf_vf_vf(v0, v1), v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_5vf(
    v0: VFloat,
    v1: VFloat,
    v2: VFloat,
    v3: VFloat,
    v4: VFloat
) -> VFloat {
    vadd_vf_4vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_6vf(
    v0: VFloat,
    v1: VFloat,
    v2: VFloat,
    v3: VFloat,
    v4: VFloat,
    v5: VFloat
) -> VFloat {
    vadd_vf_5vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4, v5)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vf_7vf(
    v0: VFloat,
    v1: VFloat,
    v2: VFloat,
    v3: VFloat,
    v4: VFloat,
    v5: VFloat,
    v6: VFloat
) -> VFloat {
    vadd_vf_6vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4, v5, v6)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_3vf(v0: VFloat, v1: VFloat, v2: VFloat) -> VFloat {
    vsub_vf_vf_vf(vsub_vf_vf_vf(v0, v1), v2)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_4vf(v0: VFloat, v1: VFloat, v2: VFloat, v3: VFloat) -> VFloat {
    vsub_vf_3vf(vsub_vf_vf_vf(v0, v1), v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_5vf(
    v0: VFloat,
    v1: VFloat,
    v2: VFloat,
    v3: VFloat,
    v4: VFloat
) -> VFloat {
    vsub_vf_4vf(vsub_vf_vf_vf(v0, v1), v2, v3, v4)
}

#[inline(always)]
pub(crate) unsafe fn dfneg_vf2_vf2(x: VFloat2) -> VFloat2 {
    vcast_vf2_vf_vf(vneg_vf_vf(vf2getx_vf_vf2(x)), vneg_vf_vf(vf2gety_vf_vf2(x)))
}

#[inline(always)]
pub(crate) unsafe fn dfabs_vf2_vf2(x: VFloat2) -> VFloat2 {
    vcast_vf2_vf_vf(
        vabs_vf_vf(vf2getx_vf_vf2(x)),
        vreinterpret_vf_vm(
            vxor_vm_vm_vm(
                vreinterpret_vm_vf(vf2gety_vf_vf2(x)),
                vand_vm_vm_vm(
                    vreinterpret_vm_vf(vf2getx_vf_vf2(x)),
                    vreinterpret_vm_vf(vcast_vf_f(-0.0))
                )
            )
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn dfnormalize_vf2_vf2(t: VFloat2) -> VFloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(t), vf2gety_vf_vf2(t));
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(vf2getx_vf_vf2(t), s), vf2gety_vf_vf2(t)))
}

#[inline(always)]
pub(crate) unsafe fn dfscale_vf2_vf2_vf(d: VFloat2, s: VFloat) -> VFloat2 {
    vf2setxy_vf2_vf_vf(vmul_vf_vf_vf(vf2getx_vf_vf2(d), s), vmul_vf_vf_vf(vf2gety_vf_vf2(d), s))
}

#[inline(always)]
pub(crate) unsafe fn dfadd_vf2_vf_vf(x: VFloat, y: VFloat) -> VFloat2 {
    let s = vadd_vf_vf_vf(x, y);
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(x, s), y))
}

#[inline(always)]
pub(crate) unsafe fn dfadd2_vf2_vf_vf(x: VFloat, y: VFloat) -> VFloat2 {
    let s = vadd_vf_vf_vf(x, y);
    let v = vsub_vf_vf_vf(s, x);
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(x, vsub_vf_vf_vf(s, v)), vsub_vf_vf_vf(y, v)))
}

#[inline(always)]
pub(crate) unsafe fn dfadd2_vf2_vf_vf2(x: VFloat, y: VFloat2) -> VFloat2 {
    let s = vadd_vf_vf_vf(x, vf2getx_vf_vf2(y));
    let v = vsub_vf_vf_vf(s, x);
    vf2setxy_vf2_vf_vf(
        s,
        vadd_vf_vf_vf(
            vadd_vf_vf_vf(
                vsub_vf_vf_vf(x, vsub_vf_vf_vf(s, v)),
                vsub_vf_vf_vf(vf2getx_vf_vf2(y), v)
            ),
            vf2gety_vf_vf2(y)
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn dfadd_vf2_vf2_vf(x: VFloat2, y: VFloat) -> VFloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), y);
    vf2setxy_vf2_vf_vf(s, vadd_vf_3vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), s), y, vf2gety_vf_vf2(x)))
}

#[inline(always)]
pub(crate) unsafe fn dfsub_vf2_vf2_vf(x: VFloat2, y: VFloat) -> VFloat2 {
    let s = vsub_vf_vf_vf(vf2getx_vf_vf2(x), y);
    vf2setxy_vf2_vf_vf(
        s,
        vadd_vf_vf_vf(vsub_vf_vf_vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), s), y), vf2gety_vf_vf2(x))
    )
}

#[inline(always)]
pub(crate) unsafe fn dfadd2_vf2_vf2_vf(x: VFloat2, y: VFloat) -> VFloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), y);
    let v = vsub_vf_vf_vf(s, vf2getx_vf_vf2(x));
    let t = vadd_vf_vf_vf(
        vsub_vf_vf_vf(vf2getx_vf_vf2(x), vsub_vf_vf_vf(s, v)),
        vsub_vf_vf_vf(y, v)
    );
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(t, vf2gety_vf_vf2(x)))
}

#[inline(always)]
pub(crate) unsafe fn dfadd_vf2_vf_vf2(x: VFloat, y: VFloat2) -> VFloat2 {
    let s = vadd_vf_vf_vf(x, vf2getx_vf_vf2(y));
    vf2setxy_vf2_vf_vf(s, vadd_vf_3vf(vsub_vf_vf_vf(x, s), vf2getx_vf_vf2(y), vf2gety_vf_vf2(y)))
}

#[inline(always)]
pub(crate) unsafe fn dfadd_vf2_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat2 {
    // |x| >= |y|
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    vf2setxy_vf2_vf_vf(
        s,
        vadd_vf_4vf(
            vsub_vf_vf_vf(vf2getx_vf_vf2(x), s),
            vf2getx_vf_vf2(y),
            vf2gety_vf_vf2(x),
            vf2gety_vf_vf2(y)
        )
    )
}

#[inline(always)]
pub(crate) unsafe fn dfadd2_vf2_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let v = vsub_vf_vf_vf(s, vf2getx_vf_vf2(x));
    let t = vadd_vf_vf_vf(
        vsub_vf_vf_vf(vf2getx_vf_vf2(x), vsub_vf_vf_vf(s, v)),
        vsub_vf_vf_vf(vf2getx_vf_vf2(y), v)
    );
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(t, vadd_vf_vf_vf(vf2gety_vf_vf2(x), vf2gety_vf_vf2(y))))
}

#[inline(always)]
pub(crate) unsafe fn dfsub_vf2_vf_vf(x: VFloat, y: VFloat) -> VFloat2 {
    // |x| >= |y|
    let s = vsub_vf_vf_vf(x, y);
    vf2setxy_vf2_vf_vf(s, vsub_vf_vf_vf(vsub_vf_vf_vf(x, s), y))
}

#[inline(always)]
pub(crate) unsafe fn dfsub_vf2_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat2 {
    // |x| >= |y|
    let s = vsub_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let mut t = vsub_vf_vf_vf(vf2getx_vf_vf2(x), s);
    t = vsub_vf_vf_vf(t, vf2getx_vf_vf2(y));
    t = vadd_vf_vf_vf(t, vf2gety_vf_vf2(x));
    vf2setxy_vf2_vf_vf(s, vsub_vf_vf_vf(t, vf2gety_vf_vf2(y)))
}

// #ifdef ENABLE_FMA_SP
#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfdiv_vf2_vf2_vf2(n: VFloat2, d: VFloat2) -> VFloat2 {
    let t = vrec_vf_vf(vf2getx_vf_vf2(d));
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(n), t);
    let u = vfmapn_vf_vf_vf_vf(t, vf2getx_vf_vf2(n), s);
    let v = vfmanp_vf_vf_vf_vf(
        vf2gety_vf_vf2(d),
        t,
        vfmanp_vf_vf_vf_vf(vf2getx_vf_vf2(d), t, vcast_vf_f(1.0))
    );
    vf2setxy_vf2_vf_vf(s, vfma_vf_vf_vf_vf(s, v, vfma_vf_vf_vf_vf(vf2gety_vf_vf2(n), t, u)))
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfmul_vf2_vf_vf(x: VFloat, y: VFloat) -> VFloat2 {
    let s = vmul_vf_vf_vf(x, y);
    vf2setxy_vf2_vf_vf(s, vfmapn_vf_vf_vf_vf(x, y, s))
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfsqu_vf2_vf2(x: VFloat2) -> VFloat2 {
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));
    vf2setxy_vf2_vf_vf(
        s,
        vfma_vf_vf_vf_vf(
            vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x)),
            vf2gety_vf_vf2(x),
            vfmapn_vf_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x), s)
        )
    )
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfsqu_vf_vf2(x: VFloat2) -> VFloat {
    vfma_vf_vf_vf_vf(
        vf2getx_vf_vf2(x),
        vf2getx_vf_vf2(x),
        vadd_vf_vf_vf(
            vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(x)),
            vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(x))
        )
    )
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfmul_vf2_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat2 {
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    vf2setxy_vf2_vf_vf(
        s,
        vfma_vf_vf_vf_vf(
            vf2getx_vf_vf2(x),
            vf2gety_vf_vf2(y),
            vfma_vf_vf_vf_vf(
                vf2gety_vf_vf2(x),
                vf2getx_vf_vf2(y),
                vfmapn_vf_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y), s)
            )
        )
    )
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfmul_vf_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat {
    vfma_vf_vf_vf_vf(
        vf2getx_vf_vf2(x),
        vf2getx_vf_vf2(y),
        vfma_vf_vf_vf_vf(
            vf2gety_vf_vf2(x),
            vf2getx_vf_vf2(y),
            vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(y))
        )
    )
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfmul_vf2_vf2_vf(x: VFloat2, y: VFloat) -> VFloat2 {
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), y);
    vf2setxy_vf2_vf_vf(
        s,
        vfma_vf_vf_vf_vf(vf2gety_vf_vf2(x), y, vfmapn_vf_vf_vf_vf(vf2getx_vf_vf2(x), y, s))
    )
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfrec_vf2_vf(d: VFloat) -> VFloat2 {
    let s = vrec_vf_vf(d);
    vf2setxy_vf2_vf_vf(s, vmul_vf_vf_vf(s, vfmanp_vf_vf_vf_vf(d, s, vcast_vf_f(1.0))))
}

#[inline(always)]
#[cfg(target_feature = "fma")]
pub(crate) unsafe fn dfrec_vf2_vf2(d: VFloat2) -> VFloat2 {
    let s = vrec_vf_vf(vf2getx_vf_vf2(d));
    vf2setxy_vf2_vf_vf(
        s,
        vmul_vf_vf_vf(
            s,
            vfmanp_vf_vf_vf_vf(
                vf2gety_vf_vf2(d),
                s,
                vfmanp_vf_vf_vf_vf(vf2getx_vf_vf2(d), s, vcast_vf_f(1.0))
            )
        )
    )
}
// #else
#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfdiv_vf2_vf2_vf2(n: VFloat2, d: VFloat2) -> VFloat2 {
    let t = vrec_vf_vf(vf2getx_vf_vf2(d));
    let dh = vupper_vf_vf(vf2getx_vf_vf2(d));
    let dl = vsub_vf_vf_vf(vf2getx_vf_vf2(d), dh);
    let th = vupper_vf_vf(t);
    let tl = vsub_vf_vf_vf(t, th);
    let nhh = vupper_vf_vf(vf2getx_vf_vf2(n));
    let nhl = vsub_vf_vf_vf(vf2getx_vf_vf2(n), nhh);

    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(n), t);

    let mut w = vcast_vf_f(-1.0);
    w = vmla_vf_vf_vf_vf(dh, th, w);
    w = vmla_vf_vf_vf_vf(dh, tl, w);
    w = vmla_vf_vf_vf_vf(dl, th, w);
    w = vmla_vf_vf_vf_vf(dl, tl, w);
    w = vneg_vf_vf(w);

    let mut u = vmla_vf_vf_vf_vf(nhh, th, vneg_vf_vf(s));
    u = vmla_vf_vf_vf_vf(nhh, tl, u);
    u = vmla_vf_vf_vf_vf(nhl, th, u);
    u = vmla_vf_vf_vf_vf(nhl, tl, u);
    u = vmla_vf_vf_vf_vf(s, w, u);

    vf2setxy_vf2_vf_vf(
        s,
        vmla_vf_vf_vf_vf(
            t,
            vsub_vf_vf_vf(vf2gety_vf_vf2(n), vmul_vf_vf_vf(s, vf2gety_vf_vf2(d))),
            u
        )
    )
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfmul_vf2_vf_vf(x: VFloat, y: VFloat) -> VFloat2 {
    let xh = vupper_vf_vf(x);
    let xl = vsub_vf_vf_vf(x, xh);
    let yh = vupper_vf_vf(y);
    let yl = vsub_vf_vf_vf(y, yh);

    let s = vmul_vf_vf_vf(x, y);
    let mut t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(xl, yh, t);
    t = vmla_vf_vf_vf_vf(xh, yl, t);
    t = vmla_vf_vf_vf_vf(xl, yl, t);

    vf2setxy_vf2_vf_vf(s, t)
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfmul_vf2_vf2_vf(x: VFloat2, y: VFloat) -> VFloat2 {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
    let yh = vupper_vf_vf(y);
    let yl = vsub_vf_vf_vf(y, yh);

    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), y);
    let mut t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(xl, yh, t);
    t = vmla_vf_vf_vf_vf(xh, yl, t);
    t = vmla_vf_vf_vf_vf(xl, yl, t);
    t = vmla_vf_vf_vf_vf(vf2gety_vf_vf2(x), y, t);

    vf2setxy_vf2_vf_vf(s, t)
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfmul_vf2_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat2 {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
    let yh = vupper_vf_vf(vf2getx_vf_vf2(y));
    let yl = vsub_vf_vf_vf(vf2getx_vf_vf2(y), yh);

    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let mut t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(xl, yh, t);
    t = vmla_vf_vf_vf_vf(xh, yl, t);
    t = vmla_vf_vf_vf_vf(xl, yl, t);
    t = vmla_vf_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(y), t);
    t = vmla_vf_vf_vf_vf(vf2gety_vf_vf2(x), vf2getx_vf_vf2(y), t);

    vf2setxy_vf2_vf_vf(s, t)
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfmul_vf_vf2_vf2(x: VFloat2, y: VFloat2) -> VFloat {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
    let yh = vupper_vf_vf(vf2getx_vf_vf2(y));
    let yl = vsub_vf_vf_vf(vf2getx_vf_vf2(y), yh);

    vadd_vf_6vf(
        vmul_vf_vf_vf(vf2gety_vf_vf2(x), yh),
        vmul_vf_vf_vf(xh, vf2gety_vf_vf2(y)),
        vmul_vf_vf_vf(xl, yl),
        vmul_vf_vf_vf(xh, yl),
        vmul_vf_vf_vf(xl, yh),
        vmul_vf_vf_vf(xh, yh)
    )
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfsqu_vf2_vf2(x: VFloat2) -> VFloat2 {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);

    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));
    let mut t = vmla_vf_vf_vf_vf(xh, xh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(vadd_vf_vf_vf(xh, xh), xl, t);
    t = vmla_vf_vf_vf_vf(xl, xl, t);
    t = vmla_vf_vf_vf_vf(vf2getx_vf_vf2(x), vadd_vf_vf_vf(vf2gety_vf_vf2(x), vf2gety_vf_vf2(x)), t);

    vf2setxy_vf2_vf_vf(s, t)
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfsqu_vf_vf2(x: VFloat2) -> VFloat {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);

    vadd_vf_5vf(
        vmul_vf_vf_vf(xh, vf2gety_vf_vf2(x)),
        vmul_vf_vf_vf(xh, vf2gety_vf_vf2(x)),
        vmul_vf_vf_vf(xl, xl),
        vadd_vf_vf_vf(vmul_vf_vf_vf(xh, xl), vmul_vf_vf_vf(xh, xl)),
        vmul_vf_vf_vf(xh, xh)
    )
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfrec_vf2_vf(d: VFloat) -> VFloat2 {
    let t = vrec_vf_vf(d);
    let dh = vupper_vf_vf(d);
    let dl = vsub_vf_vf_vf(d, dh);
    let th = vupper_vf_vf(t);
    let tl = vsub_vf_vf_vf(t, th);

    let mut u = vcast_vf_f(-1.0);
    u = vmla_vf_vf_vf_vf(dh, th, u);
    u = vmla_vf_vf_vf_vf(dh, tl, u);
    u = vmla_vf_vf_vf_vf(dl, th, u);
    u = vmla_vf_vf_vf_vf(dl, tl, u);

    vf2setxy_vf2_vf_vf(t, vmul_vf_vf_vf(vneg_vf_vf(t), u))
}

#[inline(always)]
#[cfg(not(target_feature = "fma"))]
pub(crate) unsafe fn dfrec_vf2_vf2(d: VFloat2) -> VFloat2 {
    let t = vrec_vf_vf(vf2getx_vf_vf2(d));
    let dh = vupper_vf_vf(vf2getx_vf_vf2(d));
    let dl = vsub_vf_vf_vf(vf2getx_vf_vf2(d), dh);
    let th = vupper_vf_vf(t);
    let tl = vsub_vf_vf_vf(t, th);

    let mut u = vcast_vf_f(-1.0);
    u = vmla_vf_vf_vf_vf(dh, th, u);
    u = vmla_vf_vf_vf_vf(dh, tl, u);
    u = vmla_vf_vf_vf_vf(dl, th, u);
    u = vmla_vf_vf_vf_vf(dl, tl, u);
    u = vmla_vf_vf_vf_vf(vf2gety_vf_vf2(d), t, u);

    vf2setxy_vf2_vf_vf(t, vmul_vf_vf_vf(vneg_vf_vf(t), u))
}

#[inline(always)]
pub(crate) unsafe fn dfsqrt_vf2_vf(d: VFloat) -> VFloat2 {
    let t = vsqrt_vf_vf(d);
    dfscale_vf2_vf2_vf(
        dfmul_vf2_vf2_vf2(dfadd2_vf2_vf_vf2(d, dfmul_vf2_vf_vf(t, t)), dfrec_vf2_vf(t)),
        vcast_vf_f(0.5f32)
    )
}

#[inline(always)]
pub(crate) unsafe fn dfsqrt_vf2_vf2(d: VFloat2) -> VFloat2 {
    let t = vsqrt_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)));
    dfscale_vf2_vf2_vf(
        dfmul_vf2_vf2_vf2(dfadd2_vf2_vf2_vf2(d, dfmul_vf2_vf_vf(t, t)), dfrec_vf2_vf(t)),
        vcast_vf_f(0.5)
    )
}
