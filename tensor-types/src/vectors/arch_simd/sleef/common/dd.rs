#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::arch_simd::sleef::arch::helper_avx2 as helper;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "avx2")
))]
use crate::arch_simd::sleef::arch::helper_sse as helper;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::arch_simd::sleef::arch::helper_neon as helper;

use helper::{
    vadd_vd_vd_vd, vcast_vd_d, vmul_vd_vd_vd, vneg_vd_vd, vsel_vd_vo_vd_vd, vsqrt_vd_vd,
    vsub_vd_vd_vd,
};

use crate::sleef_types::{VDouble, Vopmask};

#[derive(Clone, Copy)]
pub(crate) struct VDouble2 {
    pub(crate) x: VDouble,
    pub(crate) y: VDouble,
}

pub(crate) fn vd2getx_vd_vd2(v: VDouble2) -> VDouble {
    v.x
}
pub(crate) fn vd2gety_vd_vd2(v: VDouble2) -> VDouble {
    v.y
}

#[inline(always)]
pub(crate) unsafe fn vd2setxy_vd2_vd_vd(x: VDouble, y: VDouble) -> VDouble2 {
    VDouble2 { x, y }
}

#[inline(always)]
pub(crate) unsafe fn vd2setx_vd2_vd2_vd(mut v: VDouble2, d: VDouble) -> VDouble2 {
    v.x = d;
    v
}

#[inline(always)]
pub(crate) unsafe fn vd2sety_vd2_vd2_vd(mut v: VDouble2, d: VDouble) -> VDouble2 {
    v.y = d;
    v
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd2_vd_vd(h: VDouble, l: VDouble) -> VDouble2 {
        vd2setxy_vd2_vd_vd(h, l)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd2_d_d(h: f64, l: f64) -> VDouble2 {
        vd2setxy_vd2_vd_vd(vcast_vd_d(h), vcast_vd_d(l))
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd2_vo_vd2_vd2(m: Vopmask, x: VDouble2, y: VDouble2) -> VDouble2 {
        vd2setxy_vd2_vd_vd(
        vsel_vd_vo_vd_vd(m, vd2getx_vd_vd2(x), vd2getx_vd_vd2(y)),
        vsel_vd_vo_vd_vd(m, vd2gety_vd_vd2(x), vd2gety_vd_vd2(y)),
    )
}

#[inline(always)]
pub(crate) unsafe fn vadd_vd_3vd(v0: VDouble, v1: VDouble, v2: VDouble) -> VDouble {
        vadd_vd_vd_vd(vadd_vd_vd_vd(v0, v1), v2)
}

#[inline(always)]
pub(crate) unsafe fn vadd_vd_4vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble) -> VDouble {
        vadd_vd_3vd(vadd_vd_vd_vd(v0, v1), v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn ddneg_vd2_vd2(x: VDouble2) -> VDouble2 {
        vcast_vd2_vd_vd(vneg_vd_vd(vd2getx_vd_vd2(x)), vneg_vd_vd(vd2gety_vd_vd2(x)))
}

#[inline(always)]
pub(crate) unsafe fn ddnormalize_vd2_vd2(t: VDouble2) -> VDouble2 {
        let s = vadd_vd_vd_vd(vd2getx_vd_vd2(t), vd2gety_vd_vd2(t));
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_vd_vd(vsub_vd_vd_vd(vd2getx_vd_vd2(t), s), vd2gety_vd_vd2(t)),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddscale_vd2_vd2_vd(d: VDouble2, s: VDouble) -> VDouble2 {
        vd2setxy_vd2_vd_vd(
        vmul_vd_vd_vd(vd2getx_vd_vd2(d), s),
        vmul_vd_vd_vd(vd2gety_vd_vd2(d), s),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddadd_vd2_vd_vd(x: VDouble, y: VDouble) -> VDouble2 {
        let s = vadd_vd_vd_vd(x, y);
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(vsub_vd_vd_vd(x, s), y))
}

#[inline(always)]
pub(crate) unsafe fn ddadd2_vd2_vd_vd(x: VDouble, y: VDouble) -> VDouble2 {
        let s = vadd_vd_vd_vd(x, y);
    let v = vsub_vd_vd_vd(s, x);
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(s, v)), vsub_vd_vd_vd(y, v)),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddadd_vd2_vd2_vd(x: VDouble2, y: VDouble) -> VDouble2 {
        let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), y);
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_3vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), s), y, vd2gety_vd_vd2(x)),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddsub_vd2_vd2_vd(x: VDouble2, y: VDouble) -> VDouble2 {
        let s = vsub_vd_vd_vd(vd2getx_vd_vd2(x), y);
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_vd_vd(
            vsub_vd_vd_vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), s), y),
            vd2gety_vd_vd2(x),
        ),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddadd2_vd2_vd2_vd(x: VDouble2, y: VDouble) -> VDouble2 {
        let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), y);
    let v = vsub_vd_vd_vd(s, vd2getx_vd_vd2(x));
    let w = vadd_vd_vd_vd(
        vsub_vd_vd_vd(vd2getx_vd_vd2(x), vsub_vd_vd_vd(s, v)),
        vsub_vd_vd_vd(y, v),
    );
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(w, vd2gety_vd_vd2(x)))
}

#[inline(always)]
pub(crate) unsafe fn ddadd_vd2_vd_vd2(x: VDouble, y: VDouble2) -> VDouble2 {
        let s = vadd_vd_vd_vd(x, vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_3vd(vsub_vd_vd_vd(x, s), vd2getx_vd_vd2(y), vd2gety_vd_vd2(y)),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddadd2_vd2_vd_vd2(x: VDouble, y: VDouble2) -> VDouble2 {
        let s = vadd_vd_vd_vd(x, vd2getx_vd_vd2(y));
    let v = vsub_vd_vd_vd(s, x);
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_vd_vd(
            vadd_vd_vd_vd(
                vsub_vd_vd_vd(x, vsub_vd_vd_vd(s, v)),
                vsub_vd_vd_vd(vd2getx_vd_vd2(y), v),
            ),
            vd2gety_vd_vd2(y),
        ),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddadd_vd2_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble2 {
        let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_4vd(
            vsub_vd_vd_vd(vd2getx_vd_vd2(x), s),
            vd2getx_vd_vd2(y),
            vd2gety_vd_vd2(x),
            vd2gety_vd_vd2(y),
        ),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddadd2_vd2_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble2 {
        let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    let v = vsub_vd_vd_vd(s, vd2getx_vd_vd2(x));
    let t = vadd_vd_vd_vd(
        vsub_vd_vd_vd(vd2getx_vd_vd2(x), vsub_vd_vd_vd(s, v)),
        vsub_vd_vd_vd(vd2getx_vd_vd2(y), v),
    );
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_vd_vd(t, vadd_vd_vd_vd(vd2gety_vd_vd2(x), vd2gety_vd_vd2(y))),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddsub_vd2_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble2 {
        let s = vsub_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    let mut t = vsub_vd_vd_vd(vd2getx_vd_vd2(x), s);
    t = vsub_vd_vd_vd(t, vd2getx_vd_vd2(y));
    t = vadd_vd_vd_vd(t, vd2gety_vd_vd2(x));
    vd2setxy_vd2_vd_vd(s, vsub_vd_vd_vd(t, vd2gety_vd_vd2(y)))
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn dddiv_vd2_vd2_vd2(n: VDouble2, d: VDouble2) -> VDouble2 {
    
    use helper::{
        vfma_vd_vd_vd_vd, vfmanp_vd_vd_vd_vd, vfmapn_vd_vd_vd_vd, vrec_vd_vd,
    };
    let t = vrec_vd_vd(vd2getx_vd_vd2(d));
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(n), t);
    let u = vfmapn_vd_vd_vd_vd(t, vd2getx_vd_vd2(n), s);
    let v = vfmanp_vd_vd_vd_vd(
        vd2gety_vd_vd2(d),
        t,
        vfmanp_vd_vd_vd_vd(vd2getx_vd_vd2(d), t, vcast_vd_d(1.0)),
    );
    vd2setxy_vd2_vd_vd(
        s,
        vfma_vd_vd_vd_vd(s, v, vfma_vd_vd_vd_vd(vd2gety_vd_vd2(n), t, u)),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd2_vd_vd(x: VDouble, y: VDouble) -> VDouble2 {
    
    use helper::vfmapn_vd_vd_vd_vd;
    let s = vmul_vd_vd_vd(x, y);
    vd2setxy_vd2_vd_vd(s, vfmapn_vd_vd_vd_vd(x, y, s))
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddsqu_vd2_vd2(x: VDouble2) -> VDouble2 {
    
    use helper::{vfma_vd_vd_vd_vd, vfmapn_vd_vd_vd_vd};
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));
    vd2setxy_vd2_vd_vd(
        s,
        vfma_vd_vd_vd_vd(
            vadd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x)),
            vd2gety_vd_vd2(x),
            vfmapn_vd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x), s),
        ),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd2_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble2 {
    
    use helper::{vfma_vd_vd_vd_vd, vfmapn_vd_vd_vd_vd};
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(
        s,
        vfma_vd_vd_vd_vd(
            vd2getx_vd_vd2(x),
            vd2gety_vd_vd2(y),
            vfma_vd_vd_vd_vd(
                vd2gety_vd_vd2(x),
                vd2getx_vd_vd2(y),
                vfmapn_vd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y), s),
            ),
        ),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble {
    
    use helper::vfma_vd_vd_vd_vd;
    vfma_vd_vd_vd_vd(
        vd2getx_vd_vd2(x),
        vd2getx_vd_vd2(y),
        vfma_vd_vd_vd_vd(
            vd2gety_vd_vd2(x),
            vd2getx_vd_vd2(y),
            vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(y)),
        ),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddsqu_vd_vd2(x: VDouble2) -> VDouble {
    
    use helper::vfma_vd_vd_vd_vd;
    vfma_vd_vd_vd_vd(
        vd2getx_vd_vd2(x),
        vd2getx_vd_vd2(x),
        vadd_vd_vd_vd(
            vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(x)),
            vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(x)),
        ),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd2_vd2_vd(x: VDouble2, y: VDouble) -> VDouble2 {
    
    use helper::{vfma_vd_vd_vd_vd, vfmapn_vd_vd_vd_vd};
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), y);
    vd2setxy_vd2_vd_vd(
        s,
        vfma_vd_vd_vd_vd(
            vd2gety_vd_vd2(x),
            y,
            vfmapn_vd_vd_vd_vd(vd2getx_vd_vd2(x), y, s),
        ),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddrec_vd2_vd(d: VDouble) -> VDouble2 {
    
    use helper::{vfmanp_vd_vd_vd_vd, vrec_vd_vd};
    let s = vrec_vd_vd(d);
    vd2setxy_vd2_vd_vd(
        s,
        vmul_vd_vd_vd(s, vfmanp_vd_vd_vd_vd(d, s, vcast_vd_d(1.0))),
    )
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn ddrec_vd2_vd2(d: VDouble2) -> VDouble2 {
    
    use helper::{vfmanp_vd_vd_vd_vd, vrec_vd_vd};
    let s = vrec_vd_vd(vd2getx_vd_vd2(d));
    vd2setxy_vd2_vd_vd(
        s,
        vmul_vd_vd_vd(
            s,
            vfmanp_vd_vd_vd_vd(
                vd2gety_vd_vd2(d),
                s,
                vfmanp_vd_vd_vd_vd(vd2getx_vd_vd2(d), s, vcast_vd_d(1.0)),
            ),
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vupper_vd_vd(d: VDouble) -> VDouble {
    use helper::{vreinterpret_vd_vm, vand_vm_vm_vm, vcast_vm_i_i, vreinterpret_vm_vd};
    vreinterpret_vd_vm(
        vand_vm_vm_vm(
            vreinterpret_vm_vd(d),
            vcast_vm_i_i(0xffffffffu32 as i32, 0xf8000000u32 as i32)
        )
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vsub_vd_3vd(v0: VDouble, v1: VDouble, v2: VDouble) -> VDouble {
    vsub_vd_vd_vd(vsub_vd_vd_vd(v0, v1), v2)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vsub_vd_4vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble) -> VDouble {
    vsub_vd_3vd(vsub_vd_vd_vd(v0, v1), v2, v3)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vsub_vd_5vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble, v4: VDouble) -> VDouble {
    vsub_vd_4vd(vsub_vd_vd_vd(v0, v1), v2, v3, v4)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vsub_vd_6vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble, v4: VDouble, v5: VDouble) -> VDouble {
    vsub_vd_5vd(vsub_vd_vd_vd(v0, v1), v2, v3, v4, v5)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vadd_vd_5vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble, v4: VDouble) -> VDouble {
    vadd_vd_4vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vadd_vd_6vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble, v4: VDouble, v5: VDouble) -> VDouble {
    vadd_vd_5vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4, v5)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn vadd_vd_7vd(v0: VDouble, v1: VDouble, v2: VDouble, v3: VDouble, v4: VDouble, v5: VDouble, v6: VDouble) -> VDouble {
    vadd_vd_6vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4, v5, v6)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn dddiv_vd2_vd2_vd2(n: VDouble2, d: VDouble2) -> VDouble2 {
    use helper::{vrec_vd_vd, vmla_vd_vd_vd_vd};
        let t = vrec_vd_vd(vd2getx_vd_vd2(d));

        let dh = vupper_vd_vd(vd2getx_vd_vd2(d));
    let dl = vsub_vd_vd_vd(vd2getx_vd_vd2(d), dh);
    let th = vupper_vd_vd(t);
    let tl = vsub_vd_vd_vd(t, th);
    let nhh = vupper_vd_vd(vd2getx_vd_vd2(n));
    let nhl = vsub_vd_vd_vd(vd2getx_vd_vd2(n), nhh);

    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(n), t);

    let u = vadd_vd_5vd(
        vsub_vd_vd_vd(vmul_vd_vd_vd(nhh, th), s),
        vmul_vd_vd_vd(nhh, tl),
        vmul_vd_vd_vd(nhl, th),
        vmul_vd_vd_vd(nhl, tl),
        vmul_vd_vd_vd(
            s,
            vsub_vd_5vd(
                vcast_vd_d(1.0),
                vmul_vd_vd_vd(dh, th),
                vmul_vd_vd_vd(dh, tl),
                vmul_vd_vd_vd(dl, th),
                vmul_vd_vd_vd(dl, tl),
            ),
        ),
    );

    vd2setxy_vd2_vd_vd(
        s,
        vmla_vd_vd_vd_vd(
            t,
            vsub_vd_vd_vd(vd2gety_vd_vd2(n), vmul_vd_vd_vd(s, vd2gety_vd_vd2(d))),
            u,
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd2_vd_vd(x: VDouble, y: VDouble) -> VDouble2 {
        let xh = vupper_vd_vd(x);
    let xl = vsub_vd_vd_vd(x, xh);
    let yh = vupper_vd_vd(y);
    let yl = vsub_vd_vd_vd(y, yh);

    let s = vmul_vd_vd_vd(x, y);
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_5vd(
            vmul_vd_vd_vd(xh, yh),
            vneg_vd_vd(s),
            vmul_vd_vd_vd(xl, yh),
            vmul_vd_vd_vd(xh, yl),
            vmul_vd_vd_vd(xl, yl),
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd2_vd2_vd(x: VDouble2, y: VDouble) -> VDouble2 {
        let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);
    let yh = vupper_vd_vd(y);
    let yl = vsub_vd_vd_vd(y, yh);

    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), y);
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_6vd(
            vmul_vd_vd_vd(xh, yh),
            vneg_vd_vd(s),
            vmul_vd_vd_vd(xl, yh),
            vmul_vd_vd_vd(xh, yl),
            vmul_vd_vd_vd(xl, yl),
            vmul_vd_vd_vd(vd2gety_vd_vd2(x), y),
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd2_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble2 {
        let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);
    let yh = vupper_vd_vd(vd2getx_vd_vd2(y));
    let yl = vsub_vd_vd_vd(vd2getx_vd_vd2(y), yh);

    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_7vd(
            vmul_vd_vd_vd(xh, yh),
            vneg_vd_vd(s),
            vmul_vd_vd_vd(xl, yh),
            vmul_vd_vd_vd(xh, yl),
            vmul_vd_vd_vd(xl, yl),
            vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(y)),
            vmul_vd_vd_vd(vd2gety_vd_vd2(x), vd2getx_vd_vd2(y)),
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddmul_vd_vd2_vd2(x: VDouble2, y: VDouble2) -> VDouble {
        let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);
    let yh = vupper_vd_vd(vd2getx_vd_vd2(y));
    let yl = vsub_vd_vd_vd(vd2getx_vd_vd2(y), yh);

    vadd_vd_6vd(
        vmul_vd_vd_vd(vd2gety_vd_vd2(x), yh),
        vmul_vd_vd_vd(xh, vd2gety_vd_vd2(y)),
        vmul_vd_vd_vd(xl, yl),
        vmul_vd_vd_vd(xh, yl),
        vmul_vd_vd_vd(xl, yh),
        vmul_vd_vd_vd(xh, yh),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddsqu_vd2_vd2(x: VDouble2) -> VDouble2 {
        let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);

    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));
    vd2setxy_vd2_vd_vd(
        s,
        vadd_vd_5vd(
            vmul_vd_vd_vd(xh, xh),
            vneg_vd_vd(s),
            vmul_vd_vd_vd(vadd_vd_vd_vd(xh, xh), xl),
            vmul_vd_vd_vd(xl, xl),
            vmul_vd_vd_vd(
                vd2getx_vd_vd2(x),
                vadd_vd_vd_vd(vd2gety_vd_vd2(x), vd2gety_vd_vd2(x)),
            ),
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddsqu_vd_vd2(x: VDouble2) -> VDouble {
        let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);

    vadd_vd_5vd(
        vmul_vd_vd_vd(xh, vd2gety_vd_vd2(x)),
        vmul_vd_vd_vd(xh, vd2gety_vd_vd2(x)),
        vmul_vd_vd_vd(xl, xl),
        vadd_vd_vd_vd(vmul_vd_vd_vd(xh, xl), vmul_vd_vd_vd(xh, xl)),
        vmul_vd_vd_vd(xh, xh),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddrec_vd2_vd(d: VDouble) -> VDouble2 {
    use helper::vrec_vd_vd;
        let t = vrec_vd_vd(d);

        let dh = vupper_vd_vd(d);
    let dl = vsub_vd_vd_vd(d, dh);
    let th = vupper_vd_vd(t);
    let tl = vsub_vd_vd_vd(t, th);

    vd2setxy_vd2_vd_vd(
        t,
        vmul_vd_vd_vd(
            t,
            vsub_vd_5vd(
                vcast_vd_d(1.0),
                vmul_vd_vd_vd(dh, th),
                vmul_vd_vd_vd(dh, tl),
                vmul_vd_vd_vd(dl, th),
                vmul_vd_vd_vd(dl, tl),
            ),
        ),
    )
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub(crate) unsafe fn ddrec_vd2_vd2(d: VDouble2) -> VDouble2 {
    use helper::vrec_vd_vd;
        let t = vrec_vd_vd(vd2getx_vd_vd2(d));

        let dh = vupper_vd_vd(vd2getx_vd_vd2(d));
    let dl = vsub_vd_vd_vd(vd2getx_vd_vd2(d), dh);
    let th = vupper_vd_vd(t);
    let tl = vsub_vd_vd_vd(t, th);

    vd2setxy_vd2_vd_vd(
        t,
        vmul_vd_vd_vd(
            t,
            vsub_vd_6vd(
                vcast_vd_d(1.0),
                vmul_vd_vd_vd(dh, th),
                vmul_vd_vd_vd(dh, tl),
                vmul_vd_vd_vd(dl, th),
                vmul_vd_vd_vd(dl, tl),
                vmul_vd_vd_vd(vd2gety_vd_vd2(d), t),
            ),
        ),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddsqrt_vd2_vd2(d: VDouble2) -> VDouble2 {
        let t = vsqrt_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)));
    ddscale_vd2_vd2_vd(
        ddmul_vd2_vd2_vd2(
            ddadd2_vd2_vd2_vd2(d, ddmul_vd2_vd_vd(t, t)),
            ddrec_vd2_vd(t),
        ),
        vcast_vd_d(0.5),
    )
}

#[inline(always)]
pub(crate) unsafe fn ddsqrt_vd2_vd(d: VDouble) -> VDouble2 {
        let t = vsqrt_vd_vd(d);
    ddscale_vd2_vd2_vd(
        ddmul_vd2_vd2_vd2(ddadd2_vd2_vd_vd2(d, ddmul_vd2_vd_vd(t, t)), ddrec_vd2_vd(t)),
        vcast_vd_d(0.5),
    )
}
