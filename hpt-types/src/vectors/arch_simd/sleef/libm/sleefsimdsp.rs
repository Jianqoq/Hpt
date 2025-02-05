#![allow(unused)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::arch_simd::sleef::arch::helper_aarch64 as helper;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::arch_simd::sleef::arch::helper_avx2 as helper;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "avx2")
))]
use crate::arch_simd::sleef::arch::helper_sse as helper;

use helper::{
    vabs_vf_vf, vadd_vf_vf_vf, vadd_vi2_vi2_vi2, vand_vi2_vi2_vi2, vand_vi2_vo_vi2, vand_vm_vm_vm,
    vand_vm_vo32_vm, vand_vo_vo_vo, vandnot_vi2_vi2_vi2, vandnot_vm_vm_vm, vandnot_vm_vo32_vm,
    vandnot_vo_vo_vo, vcast_vf_f, vcast_vf_vi2, vcast_vi2_i, veq_vo_vf_vf, veq_vo_vi2_vi2,
    vgather_vf_p_vi2, vge_vo_vf_vf, vgt_vi2_vi2_vi2, vgt_vo_vf_vf, vgt_vo_vi2_vi2, visinf_vo_vf,
    visnan_vo_vf, vispinf_vo_vf, vle_vo_vf_vf, vlt_vo_vf_vf, vmax_vf_vf_vf, vmin_vf_vf_vf,
    vmla_vf_vf_vf_vf, vmlanp_vf_vf_vf_vf, vmul_vf_vf_vf, vneg_vf_vf, vneg_vi2_vi2, vor_vi2_vi2_vi2,
    vor_vm_vm_vm, vor_vm_vo32_vm, vor_vo_vo_vo, vreinterpret_vf_vi2, vreinterpret_vf_vm,
    vreinterpret_vi2_vf, vreinterpret_vm_vf, vrint_vf_vf, vrint_vi2_vf, vsel_vf_vo_f_f,
    vsel_vf_vo_vf_vf, vsel_vi2_vo_vi2_vi2, vsll_vi2_vi2_i, vsra_vi2_vi2_i, vsrl_vi2_vi2_i,
    vsub_vf_vf_vf, vsub_vi2_vi2_vi2, vtestallones_i_vo32, vtruncate_vf_vf, vtruncate_vi2_vf,
    vxor_vm_vm_vm, vxor_vo_vo_vo,
};

use crate::vectors::arch_simd::sleef::{
    common::{
        df::{
            dfadd2_vf2_vf2_vf, dfadd2_vf2_vf2_vf2, dfadd2_vf2_vf_vf, dfadd2_vf2_vf_vf2,
            dfadd_vf2_vf2_vf, dfadd_vf2_vf2_vf2, dfadd_vf2_vf_vf, dfadd_vf2_vf_vf2,
            dfdiv_vf2_vf2_vf2, dfmul_vf2_vf2_vf, dfmul_vf2_vf2_vf2, dfmul_vf2_vf_vf,
            dfmul_vf_vf2_vf2, dfneg_vf2_vf2, dfnormalize_vf2_vf2, dfrec_vf2_vf, dfrec_vf2_vf2,
            dfscale_vf2_vf2_vf, dfsqrt_vf2_vf, dfsqrt_vf2_vf2, dfsqu_vf2_vf2, dfsqu_vf_vf2,
            dfsub_vf2_vf2_vf, dfsub_vf2_vf2_vf2, vcast_vf2_f_f, vcast_vf2_vf_vf, vf2getx_vf_vf2,
            vf2gety_vf_vf2, vf2setx_vf2_vf2_vf, vf2setxy_vf2_vf_vf, vf2sety_vf2_vf2_vf,
            vsel_vf2_vo_vf2_vf2, VFloat2,
        },
        estrin::{poly6, poly6_},
        misc::{
            L10_LF, L10_UF, L2_LF, L2_UF, LOG10_2, LOG1PF_BOUND, M_1_PI, PI_A2F, PI_B2F, PI_C2F,
            R_LN2_F, SLEEF_FLT_MIN, SQRT_FLT_MAX, TRIGRANGEMAX2F,
        },
    },
    table::SLEEF_REMPITABSP,
};

#[cfg(target_feature = "fma")]
use helper::vfma_vf_vf_vf_vf;

use crate::sleef_types::*;

#[inline(always)]
pub(crate) unsafe fn visnegzero_vo_vf(d: VFloat) -> Vopmask {
    veq_vo_vi2_vi2(
        vreinterpret_vi2_vf(d),
        vreinterpret_vi2_vf(vcast_vf_f(-0.0)),
    )
}

#[inline(always)]
pub(crate) unsafe fn vsignbit_vm_vf(f: VFloat) -> VMask {
    vand_vm_vm_vm(vreinterpret_vm_vf(f), vreinterpret_vm_vf(vcast_vf_f(-0.0)))
}

#[inline(always)]
pub(crate) unsafe fn vmulsign_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(x), vsignbit_vm_vf(y)))
}

#[inline(always)]
pub(crate) unsafe fn vcopysign_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vreinterpret_vf_vm(vxor_vm_vm_vm(
        vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(x)),
        vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(y)),
    ))
}

#[inline(always)]
pub(crate) unsafe fn vsignbit_vo_vf(d: VFloat) -> Vopmask {
    veq_vo_vi2_vi2(
        vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0x8000_0000u32 as i32)),
        vcast_vi2_i(0x8000_0000u32 as i32),
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi2_vf_vf_vi2_vi2(f0: VFloat, f1: VFloat, x: VInt2, y: VInt2) -> VInt2 {
    vsel_vi2_vo_vi2_vi2(vlt_vo_vf_vf(f0, f1), x, y)
}

#[inline(always)]
pub(crate) unsafe fn vilogbk_vi2_vf(d: VFloat) -> VInt2 {
    let o = vlt_vo_vf_vf(d, vcast_vf_f(5.421010862427522e-20));
    let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(vcast_vf_f(1.8446744073709552e19), d), d);

    let mut q = vand_vi2_vi2_vi2(
        vsrl_vi2_vi2_i::<23>(vreinterpret_vi2_vf(d)),
        vcast_vi2_i(0xff),
    );

    q = vsub_vi2_vi2_vi2(
        q,
        vsel_vi2_vo_vi2_vi2(o, vcast_vi2_i(64 + 0x7f), vcast_vi2_i(0x7f)),
    );

    q
}

#[inline(always)]
pub(crate) unsafe fn vilogb2k_vi2_vf(d: VFloat) -> VInt2 {
    let mut q = vreinterpret_vi2_vf(d);
    q = vsrl_vi2_vi2_i::<23>(q);
    q = vand_vi2_vi2_vi2(q, vcast_vi2_i(0xff));
    q = vsub_vi2_vi2_vi2(q, vcast_vi2_i(0x7f));
    q
}

#[inline(always)]
pub(crate) unsafe fn vpow2i_vf_vi2(q: VInt2) -> VFloat {
    vreinterpret_vf_vi2(vsll_vi2_vi2_i::<23>(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f))))
}

#[inline(always)]
pub(crate) unsafe fn vldexp_vf_vf_vi2(x: VFloat, q: VInt2) -> VFloat {
    let mut m = vsra_vi2_vi2_i::<31>(q);
    m = vsll_vi2_vi2_i::<4>(vsub_vi2_vi2_vi2(
        vsra_vi2_vi2_i::<6>(vadd_vi2_vi2_vi2(m, q)),
        m,
    ));
    let q = vsub_vi2_vi2_vi2(q, vsll_vi2_vi2_i::<2>(m));
    m = vadd_vi2_vi2_vi2(m, vcast_vi2_i(0x7f));
    m = vand_vi2_vi2_vi2(vgt_vi2_vi2_vi2(m, vcast_vi2_i(0)), m);
    let n = vgt_vi2_vi2_vi2(m, vcast_vi2_i(0xff));
    m = vor_vi2_vi2_vi2(
        vandnot_vi2_vi2_vi2(n, m),
        vand_vi2_vi2_vi2(n, vcast_vi2_i(0xff)),
    );
    let mut u = vreinterpret_vf_vi2(vsll_vi2_vi2_i::<23>(m));
    let x = vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(x, u), u), u), u);
    u = vreinterpret_vf_vi2(vsll_vi2_vi2_i::<23>(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f))));
    vmul_vf_vf_vf(x, u)
}

#[inline(always)]
pub(crate) unsafe fn vldexp2_vf_vf_vi2(d: VFloat, e: VInt2) -> VFloat {
    vmul_vf_vf_vf(
        vmul_vf_vf_vf(d, vpow2i_vf_vi2(vsra_vi2_vi2_i::<1>(e))),
        vpow2i_vf_vi2(vsub_vi2_vi2_vi2(e, vsra_vi2_vi2_i::<1>(e))),
    )
}

#[inline(always)]
pub(crate) unsafe fn vldexp3_vf_vf_vi2(d: VFloat, q: VInt2) -> VFloat {
    vreinterpret_vf_vi2(vadd_vi2_vi2_vi2(
        vreinterpret_vi2_vf(d),
        vsll_vi2_vi2_i::<23>(q),
    ))
}

#[derive(Clone, Copy)]
pub(crate) struct Fit {
    pub(crate) d: VFloat,
    pub(crate) i: VInt2,
}

#[inline(always)]
pub(crate) unsafe fn figetd_vf_di(d: Fit) -> VFloat {
    d.d
}

#[inline(always)]
pub(crate) unsafe fn figeti_vi2_di(d: Fit) -> VInt2 {
    d.i
}

#[inline(always)]
pub(crate) unsafe fn fisetdi_fi_vf_vi2(d: VFloat, i: VInt2) -> Fit {
    Fit { d, i }
}

#[derive(Clone, Copy)]
pub(crate) struct Dfit {
    pub(crate) df: VFloat2,
    pub(crate) i: VInt2,
}

#[inline(always)]
pub(crate) unsafe fn dfigetdf_vf2_dfi(d: Dfit) -> VFloat2 {
    d.df
}

#[inline(always)]
pub(crate) unsafe fn dfigeti_vi2_dfi(d: Dfit) -> VInt2 {
    d.i
}

#[inline(always)]
pub(crate) unsafe fn dfisetdfi_dfi_vf2_vi2(v: VFloat2, i: VInt2) -> Dfit {
    Dfit { df: v, i }
}

#[inline(always)]
pub(crate) unsafe fn dfisetdf_dfi_dfi_vf2(mut dfi: Dfit, v: VFloat2) -> Dfit {
    dfi.df = v;
    dfi
}

#[inline(always)]
pub(crate) unsafe fn vorsign_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    vreinterpret_vf_vm(vor_vm_vm_vm(vreinterpret_vm_vf(x), vsignbit_vm_vf(y)))
}

#[inline(always)]
pub(crate) unsafe fn rempisubf(x: VFloat) -> Fit {
    let c = vmulsign_vf_vf_vf(vcast_vf_f((1 << 23) as f32), x);

    let rint4x = vsel_vf_vo_vf_vf(
        vgt_vo_vf_vf(
            vabs_vf_vf(vmul_vf_vf_vf(vcast_vf_f(4.0), x)),
            vcast_vf_f((1 << 23) as f32),
        ),
        vmul_vf_vf_vf(vcast_vf_f(4.0), x),
        vorsign_vf_vf_vf(vsub_vf_vf_vf(vmla_vf_vf_vf_vf(vcast_vf_f(4.0), x, c), c), x),
    );

    let rintx = vsel_vf_vo_vf_vf(
        vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f((1 << 23) as f32)),
        x,
        vorsign_vf_vf_vf(vsub_vf_vf_vf(vadd_vf_vf_vf(x, c), c), x),
    );

    fisetdi_fi_vf_vi2(
        vmla_vf_vf_vf_vf(vcast_vf_f(-0.25), rint4x, x),
        vtruncate_vi2_vf(vmla_vf_vf_vf_vf(vcast_vf_f(-4.0), rintx, rint4x)),
    )
}

#[inline(always)]
pub(crate) unsafe fn rempif(a: VFloat) -> Dfit {
    let mut x: VFloat2;
    let mut y: VFloat2;
    let mut ex = vilogb2k_vi2_vf(a);

    #[cfg(target_feature = "avx512f")]
    {
        ex = vandnot_vi2_vi2_vi2(vsra_vi2_vi2_i::<31>(ex), ex);
        ex = vand_vi2_vi2_vi2(ex, vcast_vi2_i(127));
    }

    ex = vsub_vi2_vi2_vi2(ex, vcast_vi2_i(25));
    let mut q = vand_vi2_vo_vi2(vgt_vo_vi2_vi2(ex, vcast_vi2_i(90 - 25)), vcast_vi2_i(-64));
    let a = vldexp3_vf_vf_vi2(a, q);
    ex = vandnot_vi2_vi2_vi2(vsra_vi2_vi2_i::<31>(ex), ex);
    ex = vsll_vi2_vi2_i::<2>(ex);

    x = dfmul_vf2_vf_vf(a, vgather_vf_p_vi2(SLEEF_REMPITABSP.as_ptr(), ex));
    let mut di = rempisubf(vf2getx_vf_vf2(x));
    q = figeti_vi2_di(di);
    x = vf2setx_vf2_vf2_vf(x, figetd_vf_di(di));
    x = dfnormalize_vf2_vf2(x);

    y = dfmul_vf2_vf_vf(a, vgather_vf_p_vi2(SLEEF_REMPITABSP.as_ptr().add(1), ex));
    x = dfadd2_vf2_vf2_vf2(x, y);
    di = rempisubf(vf2getx_vf_vf2(x));
    q = vadd_vi2_vi2_vi2(q, figeti_vi2_di(di));
    x = vf2setx_vf2_vf2_vf(x, figetd_vf_di(di));
    x = dfnormalize_vf2_vf2(x);

    y = vcast_vf2_vf_vf(
        vgather_vf_p_vi2(SLEEF_REMPITABSP.as_ptr().add(2), ex),
        vgather_vf_p_vi2(SLEEF_REMPITABSP.as_ptr().add(3), ex),
    );
    y = dfmul_vf2_vf2_vf(y, a);
    x = dfadd2_vf2_vf2_vf2(x, y);
    x = dfnormalize_vf2_vf2(x);

    x = dfmul_vf2_vf2_vf2(
        x,
        vcast_vf2_f_f(
            3.1415927410125732422f32 * 2.0,
            -8.7422776573475857731e-8f32 * 2.0,
        ),
    );

    x = vsel_vf2_vo_vf2_vf2(
        vlt_vo_vf_vf(vabs_vf_vf(a), vcast_vf_f(0.7)),
        vcast_vf2_vf_vf(a, vcast_vf_f(0.0)),
        x,
    );

    dfisetdfi_dfi_vf2_vi2(x, q)
}

#[inline(always)]
pub(crate) unsafe fn xsinf_u1(d: VFloat) -> VFloat {
    let mut q: VInt2;
    let mut u: VFloat;
    let v: VFloat;
    let mut s: VFloat2;
    let mut t: VFloat2;
    let x: VFloat2;

    u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(M_1_PI as f32)));
    q = vrint_vi2_vf(u);
    v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2F), d);
    s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2F)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2F)));
    let g = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2F));

    if vtestallones_i_vo32(g) == 0 {
        let dfi = rempif(d);
        let mut q2 = vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(3));
        q2 = vadd_vi2_vi2_vi2(
            vadd_vi2_vi2_vi2(q2, q2),
            vsel_vi2_vo_vi2_vi2(
                vgt_vo_vf_vf(vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)), vcast_vf_f(0.0)),
                vcast_vi2_i(2),
                vcast_vi2_i(1),
            ),
        );
        q2 = vsra_vi2_vi2_i::<2>(q2);

        let o = veq_vo_vi2_vi2(
            vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(1)),
            vcast_vi2_i(1),
        );

        let mut x = vcast_vf2_vf_vf(
            vmulsign_vf_vf_vf(
                vcast_vf_f(3.1415927410125732422f32 * -0.5),
                vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)),
            ),
            vmulsign_vf_vf_vf(
                vcast_vf_f(-8.7422776573475857731e-08f32 * -0.5),
                vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)),
            ),
        );

        x = dfadd2_vf2_vf2_vf2(dfigetdf_vf2_dfi(dfi), x);
        let dfi = dfisetdf_dfi_dfi_vf2(dfi, vsel_vf2_vo_vf2_vf2(o, x, dfigetdf_vf2_dfi(dfi)));
        t = dfnormalize_vf2_vf2(dfigetdf_vf2_dfi(dfi));

        t = vf2setx_vf2_vf2_vf(
            t,
            vreinterpret_vf_vm(vor_vm_vo32_vm(
                vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d)),
                vreinterpret_vm_vf(vf2getx_vf_vf2(t)),
            )),
        );

        q = vsel_vi2_vo_vi2_vi2(g, q, q2);
        s = vsel_vf2_vo_vf2_vf2(g, s, t);
    }

    t = s;
    s = dfsqu_vf2_vf2(s);

    u = vcast_vf_f(2.6083159809786593541503e-06f32);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(-0.0001981069071916863322258f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.00833307858556509017944336f32),
    );

    x = dfadd_vf2_vf_vf2(
        vcast_vf_f(1.0),
        dfmul_vf2_vf2_vf2(
            dfadd_vf2_vf_vf(
                vcast_vf_f(-0.166666597127914428710938f32),
                vmul_vf_vf_vf(u, vf2getx_vf_vf2(s)),
            ),
            s,
        ),
    );

    u = dfmul_vf_vf2_vf2(t, x);

    u = vreinterpret_vf_vm(vxor_vm_vm_vm(
        vand_vm_vo32_vm(
            veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)),
            vreinterpret_vm_vf(vcast_vf_f(-0.0)),
        ),
        vreinterpret_vm_vf(u),
    ));

    vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), d, u)
}

#[inline(always)]
pub(crate) unsafe fn xcosf_u1(d: VFloat) -> VFloat {
    let mut q: VInt2;
    let mut u: VFloat;
    let mut s: VFloat2;
    let mut t: VFloat2;
    let x: VFloat2;

    let dq = vmla_vf_vf_vf_vf(
        vrint_vf_vf(vmla_vf_vf_vf_vf(
            d,
            vcast_vf_f(M_1_PI as f32),
            vcast_vf_f(-0.5),
        )),
        vcast_vf_f(2.0),
        vcast_vf_f(1.0),
    );
    q = vrint_vi2_vf(dq);

    s = dfadd2_vf2_vf_vf(d, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_A2F * 0.5)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_B2F * 0.5)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_C2F * 0.5)));

    let g = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2F));

    if vtestallones_i_vo32(g) == 0 {
        let dfi = rempif(d);
        let mut q2 = vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(3));
        q2 = vadd_vi2_vi2_vi2(
            vadd_vi2_vi2_vi2(q2, q2),
            vsel_vi2_vo_vi2_vi2(
                vgt_vo_vf_vf(vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)), vcast_vf_f(0.0)),
                vcast_vi2_i(8),
                vcast_vi2_i(7),
            ),
        );
        q2 = vsra_vi2_vi2_i::<1>(q2);

        let o = veq_vo_vi2_vi2(
            vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(1)),
            vcast_vi2_i(0),
        );

        let y = vsel_vf_vo_vf_vf(
            vgt_vo_vf_vf(vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)), vcast_vf_f(0.0)),
            vcast_vf_f(0.0),
            vcast_vf_f(-1.0),
        );

        let mut x = vcast_vf2_vf_vf(
            vmulsign_vf_vf_vf(vcast_vf_f(3.1415927410125732422f32 * -0.5), y),
            vmulsign_vf_vf_vf(vcast_vf_f(-8.7422776573475857731e-08f32 * -0.5), y),
        );

        x = dfadd2_vf2_vf2_vf2(dfigetdf_vf2_dfi(dfi), x);
        let dfi = dfisetdf_dfi_dfi_vf2(dfi, vsel_vf2_vo_vf2_vf2(o, x, dfigetdf_vf2_dfi(dfi)));
        t = dfnormalize_vf2_vf2(dfigetdf_vf2_dfi(dfi));

        t = vf2setx_vf2_vf2_vf(
            t,
            vreinterpret_vf_vm(vor_vm_vo32_vm(
                vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d)),
                vreinterpret_vm_vf(vf2getx_vf_vf2(t)),
            )),
        );

        q = vsel_vi2_vo_vi2_vi2(g, q, q2);
        s = vsel_vf2_vo_vf2_vf2(g, s, t);
    }

    t = s;
    s = dfsqu_vf2_vf2(s);
    u = vcast_vf_f(2.6083159809786593541503e-06);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(-0.0001981069071916863322258),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.00833307858556509017944336),
    );

    x = dfadd_vf2_vf_vf2(
        vcast_vf_f(1.0),
        dfmul_vf2_vf2_vf2(
            dfadd_vf2_vf_vf(
                vcast_vf_f(-0.166666597127914428710938),
                vmul_vf_vf_vf(u, vf2getx_vf_vf2(s)),
            ),
            s,
        ),
    );

    u = dfmul_vf_vf2_vf2(t, x);

    u = vreinterpret_vf_vm(vxor_vm_vm_vm(
        vand_vm_vo32_vm(
            veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0)),
            vreinterpret_vm_vf(vcast_vf_f(-0.0)),
        ),
        vreinterpret_vm_vf(u),
    ));

    u
}

#[inline(always)]
pub(crate) unsafe fn xtanf_u1(d: VFloat) -> VFloat {
    let q: VInt2;
    let mut u: VFloat;
    let v: VFloat;
    let mut s: VFloat2;
    let t: VFloat2;
    let mut x: VFloat2;
    let mut o: Vopmask;

    if vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2F))) != 0 {
        u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(2.0 * (M_1_PI as f32))));
        q = vrint_vi2_vf(u);
        v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2F * 0.5f32), d);
        s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2F * 0.5f32)));
        s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2F * 0.5f32)));
    } else {
        let dfi = rempif(d);
        q = dfigeti_vi2_dfi(dfi);
        s = dfigetdf_vf2_dfi(dfi);
        o = vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d));
        s = vf2setx_vf2_vf2_vf(
            s,
            vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(vf2getx_vf_vf2(s)))),
        );
        s = vf2sety_vf2_vf2_vf(
            s,
            vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(vf2gety_vf_vf2(s)))),
        );
    }

    o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1));
    let n = vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0)));

    s = vf2setx_vf2_vf2_vf(
        s,
        vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2getx_vf_vf2(s)), n)),
    );
    s = vf2sety_vf2_vf2_vf(
        s,
        vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2gety_vf_vf2(s)), n)),
    );

    t = s;
    s = dfsqu_vf2_vf2(s);
    s = dfnormalize_vf2_vf2(s);

    u = vcast_vf_f(0.00446636462584137916564941f32);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(-8.3920182078145444393158e-5f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.0109639242291450500488281f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.0212360303848981857299805f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.0540687143802642822265625f32),
    );

    x = dfadd_vf2_vf_vf(
        vcast_vf_f(0.133325666189193725585938f32),
        vmul_vf_vf_vf(u, vf2getx_vf_vf2(s)),
    );
    x = dfadd_vf2_vf_vf2(
        vcast_vf_f(1.0),
        dfmul_vf2_vf2_vf2(
            dfadd_vf2_vf_vf2(
                vcast_vf_f(0.33333361148834228515625f32),
                dfmul_vf2_vf2_vf2(s, x),
            ),
            s,
        ),
    );
    x = dfmul_vf2_vf2_vf2(t, x);

    x = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf2(x), x);

    u = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(x));

    vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), d, u)
}

#[inline(always)]
pub(crate) unsafe fn xasinf_u1(d: VFloat) -> VFloat {
    let o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f32));
    let x2 = vsel_vf_vo_vf_vf(
        o,
        vmul_vf_vf_vf(d, d),
        vmul_vf_vf_vf(
            vsub_vf_vf_vf(vcast_vf_f(1.0), vabs_vf_vf(d)),
            vcast_vf_f(0.5f32),
        ),
    );

    let mut x = vsel_vf2_vo_vf2_vf2(
        o,
        vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.0)),
        dfsqrt_vf2_vf(x2),
    );
    x = vsel_vf2_vo_vf2_vf2(
        veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f32)),
        vcast_vf2_f_f(0.0, 0.0),
        x,
    );

    let mut u = vcast_vf_f(0.4197454825e-1f32);
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.2424046025e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.4547423869e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.7495029271e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.1666677296f32));
    u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)));

    let y = dfsub_vf2_vf2_vf(
        dfsub_vf2_vf2_vf2(
            vcast_vf2_f_f(
                3.1415927410125732422f32 / 4.0,
                -8.7422776573475857731e-8f32 / 4.0,
            ),
            x,
        ),
        u,
    );

    let r = vsel_vf_vo_vf_vf(
        o,
        vadd_vf_vf_vf(u, vf2getx_vf_vf2(x)),
        vmul_vf_vf_vf(
            vadd_vf_vf_vf(vf2getx_vf_vf2(y), vf2gety_vf_vf2(y)),
            vcast_vf_f(2.0),
        ),
    );

    vmulsign_vf_vf_vf(r, d)
}

#[inline(always)]
pub(crate) unsafe fn xacosf_u1(d: VFloat) -> VFloat {
    let o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f32));
    let x2 = vsel_vf_vo_vf_vf(
        o,
        vmul_vf_vf_vf(d, d),
        vmul_vf_vf_vf(
            vsub_vf_vf_vf(vcast_vf_f(1.0), vabs_vf_vf(d)),
            vcast_vf_f(0.5f32),
        ),
    );

    let mut x = vsel_vf2_vo_vf2_vf2(
        o,
        vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.0)),
        dfsqrt_vf2_vf(x2),
    );
    x = vsel_vf2_vo_vf2_vf2(
        veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f32)),
        vcast_vf2_f_f(0.0, 0.0),
        x,
    );

    let mut u = vcast_vf_f(0.4197454825e-1f32);
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.2424046025e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.4547423869e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.7495029271e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.1666677296f32));
    u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)));

    let mut y = dfsub_vf2_vf2_vf2(
        vcast_vf2_f_f(
            3.1415927410125732422f32 / 2.0,
            -8.7422776573475857731e-8f32 / 2.0,
        ),
        dfadd_vf2_vf_vf(
            vmulsign_vf_vf_vf(vf2getx_vf_vf2(x), d),
            vmulsign_vf_vf_vf(u, d),
        ),
    );
    x = dfadd_vf2_vf2_vf(x, u);

    y = vsel_vf2_vo_vf2_vf2(o, y, dfscale_vf2_vf2_vf(x, vcast_vf_f(2.0)));

    y = vsel_vf2_vo_vf2_vf2(
        vandnot_vo_vo_vo(o, vlt_vo_vf_vf(d, vcast_vf_f(0.0))),
        dfsub_vf2_vf2_vf2(
            vcast_vf2_f_f(3.1415927410125732422f32, -8.7422776573475857731e-8f32),
            y,
        ),
        y,
    );

    vadd_vf_vf_vf(vf2getx_vf_vf2(y), vf2gety_vf_vf2(y))
}

#[inline(always)]
unsafe fn atan2kf_u1(y: VFloat2, mut x: VFloat2) -> VFloat2 {
    let mut q = vsel_vi2_vf_vf_vi2_vi2(
        vf2getx_vf_vf2(x),
        vcast_vf_f(0.0),
        vcast_vi2_i(-2),
        vcast_vi2_i(0),
    );
    let p = vlt_vo_vf_vf(vf2getx_vf_vf2(x), vcast_vf_f(0.0));
    let r = vand_vm_vo32_vm(p, vreinterpret_vm_vf(vcast_vf_f(-0.0)));
    x = vf2setx_vf2_vf2_vf(
        x,
        vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2getx_vf_vf2(x)), r)),
    );
    x = vf2sety_vf2_vf2_vf(
        x,
        vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2gety_vf_vf2(x)), r)),
    );

    q = vsel_vi2_vf_vf_vi2_vi2(
        vf2getx_vf_vf2(x),
        vf2getx_vf_vf2(y),
        vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)),
        q,
    );
    let p = vlt_vo_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let mut s = vsel_vf2_vo_vf2_vf2(p, dfneg_vf2_vf2(x), y);
    let t = vsel_vf2_vo_vf2_vf2(p, y, x);

    s = dfdiv_vf2_vf2_vf2(s, t);
    let mut t = dfsqu_vf2_vf2(s);
    t = dfnormalize_vf2_vf2(t);

    let mut u = vcast_vf_f(-0.00176397908944636583328247f32);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(0.0107900900766253471374512f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(-0.0309564601629972457885742f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(0.0577365085482597351074219f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(-0.0838950723409652709960938f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(0.109463557600975036621094f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(-0.142626821994781494140625f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(t),
        vcast_vf_f(0.199983194470405578613281f32),
    );

    t = dfmul_vf2_vf2_vf2(
        t,
        dfadd_vf2_vf_vf(
            vcast_vf_f(-0.333332866430282592773438f32),
            vmul_vf_vf_vf(u, vf2getx_vf_vf2(t)),
        ),
    );
    t = dfmul_vf2_vf2_vf2(s, dfadd_vf2_vf_vf2(vcast_vf_f(1.0), t));
    t = dfadd_vf2_vf2_vf2(
        dfmul_vf2_vf2_vf(
            vcast_vf2_f_f(1.5707963705062866211f32, -4.3711388286737928865e-8f32),
            vcast_vf_vi2(q),
        ),
        t,
    );

    t
}

#[inline(always)]
pub(crate) unsafe fn xatanf_u1(d: VFloat) -> VFloat {
    let d2 = atan2kf_u1(
        vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.0)),
        vcast_vf2_f_f(1.0, 0.0),
    );

    let r = vadd_vf_vf_vf(vf2getx_vf_vf2(d2), vf2gety_vf_vf2(d2));
    let r = vsel_vf_vo_vf_vf(
        visinf_vo_vf(d),
        vcast_vf_f(1.570796326794896557998982f32),
        r,
    );

    vmulsign_vf_vf_vf(r, d)
}

#[inline(always)]
unsafe fn visinf2_vf_vf_vf(d: VFloat, m: VFloat) -> VFloat {
    let is_inf = visinf_vo_vf(d);

    let sign = vsignbit_vm_vf(d);

    let merged = vor_vm_vm_vm(sign, vreinterpret_vm_vf(m));

    vreinterpret_vf_vm(vand_vm_vo32_vm(is_inf, merged))
}

#[inline(always)]
pub(crate) unsafe fn xatan2f_u1(y: VFloat, x: VFloat) -> VFloat {
    let o = vlt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(2.9387372783541830947e-39f32));
    let x = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(x, vcast_vf_f((1 << 24) as f32)), x);
    let y = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(y, vcast_vf_f((1 << 24) as f32)), y);

    let d = atan2kf_u1(
        vcast_vf2_vf_vf(vabs_vf_vf(y), vcast_vf_f(0.0)),
        vcast_vf2_vf_vf(x, vcast_vf_f(0.0)),
    );
    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));

    r = vmulsign_vf_vf_vf(r, x);

    r = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0.0))),
        vsub_vf_vf_vf(
            vcast_vf_f(std::f32::consts::PI / 2.0),
            visinf2_vf_vf_vf(
                x,
                vmulsign_vf_vf_vf(vcast_vf_f(std::f32::consts::PI / 2.0), x),
            ),
        ),
        r,
    );

    r = vsel_vf_vo_vf_vf(
        visinf_vo_vf(y),
        vsub_vf_vf_vf(
            vcast_vf_f(std::f32::consts::PI / 2.0),
            visinf2_vf_vf_vf(
                x,
                vmulsign_vf_vf_vf(vcast_vf_f(std::f32::consts::PI / 4.0), x),
            ),
        ),
        r,
    );

    r = vsel_vf_vo_vf_vf(
        veq_vo_vf_vf(y, vcast_vf_f(0.0)),
        vreinterpret_vf_vm(vand_vm_vo32_vm(
            vsignbit_vo_vf(x),
            vreinterpret_vm_vf(vcast_vf_f(std::f32::consts::PI)),
        )),
        r,
    );

    r = vreinterpret_vf_vm(vor_vm_vo32_vm(
        vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)),
        vreinterpret_vm_vf(vmulsign_vf_vf_vf(r, y)),
    ));

    r
}

#[inline(always)]
pub(crate) unsafe fn xsincosf_u1(d: VFloat) -> VFloat2 {
    let mut q: VInt2;
    let mut o: VMask;
    let mut u: VFloat;
    let v: VFloat;
    let mut rx: VFloat;
    let ry: VFloat;
    let mut r: VFloat2;
    let mut s: VFloat2;
    let mut t: VFloat2;
    let mut x: VFloat2;

    u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f((2.0 * M_1_PI) as f32)));
    q = vrint_vi2_vf(u);

    v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2F * 0.5), d);
    s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2F * 0.5)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2F * 0.5)));

    let g = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2F));

    if vtestallones_i_vo32(g) == 0 {
        let dfi = rempif(d);
        t = dfigetdf_vf2_dfi(dfi);
        o = vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d));
        t = vf2setx_vf2_vf2_vf(
            t,
            vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(vf2getx_vf_vf2(t)))),
        );
        q = vsel_vi2_vo_vi2_vi2(g, q, dfigeti_vi2_dfi(dfi));
        s = vsel_vf2_vo_vf2_vf2(g, s, t);
    }

    t = s;

    s = vf2setx_vf2_vf2_vf(s, dfsqu_vf_vf2(s));

    u = vcast_vf_f(-0.000195169282960705459117889);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.00833215750753879547119141),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(-0.166666537523269653320312),
    );

    u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(vf2getx_vf_vf2(s), vf2getx_vf_vf2(t)));

    x = dfadd_vf2_vf2_vf(t, u);
    rx = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(x));

    rx = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0), rx);

    u = vcast_vf_f(-2.71811842367242206819355e-07);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(2.47990446951007470488548e-05),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(-0.00138888787478208541870117),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.0416666641831398010253906),
    );
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(-0.5));

    x = dfadd_vf2_vf_vf2(vcast_vf_f(1.0), dfmul_vf2_vf_vf(vf2getx_vf_vf2(s), u));
    ry = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(x));

    o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(0));
    r = vf2setxy_vf2_vf_vf(vsel_vf_vo_vf_vf(o, rx, ry), vsel_vf_vo_vf_vf(o, ry, rx));

    o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2));
    r = vf2setx_vf2_vf2_vf(
        r,
        vreinterpret_vf_vm(vxor_vm_vm_vm(
            vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))),
            vreinterpret_vm_vf(vf2getx_vf_vf2(r)),
        )),
    );

    o = veq_vo_vi2_vi2(
        vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(2)),
        vcast_vi2_i(2),
    );
    r = vf2sety_vf2_vf2_vf(
        r,
        vreinterpret_vf_vm(vxor_vm_vm_vm(
            vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))),
            vreinterpret_vm_vf(vf2gety_vf_vf2(r)),
        )),
    );

    r
}

#[inline(always)]
pub(crate) unsafe fn xlogf_u1(d: VFloat) -> VFloat {
    let x: VFloat2;
    let mut t: VFloat;
    let x2: VFloat;

    #[cfg(not(target_feature = "avx512f"))]
    let (m, s) = {
        let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
        let d = vsel_vf_vo_vf_vf(
            o,
            vmul_vf_vf_vf(d, vcast_vf_f(((1i64 << 32) as f32) * ((1i64 << 32) as f32))),
            d,
        );
        let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32 / 0.75f32)));
        let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
        e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
        let s = dfmul_vf2_vf2_vf(
            vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-9f32),
            vcast_vf_vi2(e),
        );
        (m, s)
    };

    #[cfg(target_feature = "avx512f")]
    let (m, s) = {
        let mut e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32 / 0.75f32)));
        e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f32), e);
        let m = vgetmant_vf_vf(d);
        let s = dfmul_vf2_vf2_vf(
            vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-9f32),
            e,
        );
        (m, s)
    };

    x = dfdiv_vf2_vf2_vf2(
        dfadd2_vf2_vf_vf(vcast_vf_f(-1.0), m),
        dfadd2_vf2_vf_vf(vcast_vf_f(1.0), m),
    );
    x2 = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));

    t = vcast_vf_f(0.3027294874f32);
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.3996108174f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.666669488f32));

    let mut s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2.0)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)), t));

    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(s), vf2gety_vf_vf2(s));

    #[cfg(not(target_feature = "avx512f"))]
    {
        r = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(f32::INFINITY), r);
        r = vsel_vf_vo_vf_vf(
            vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0.0)), visnan_vo_vf(d)),
            vcast_vf_f(f32::NAN),
            r,
        );
        r = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(d, vcast_vf_f(0.0)),
            vcast_vf_f(f32::NEG_INFINITY),
            r,
        );
    }

    #[cfg(target_feature = "avx512f")]
    {
        r = vfixup_vf_vf_vf_vi2_i(
            r,
            d,
            vcast_vi2_i((4 << (2 * 4)) | (3 << (4 * 4)) | (5 << (5 * 4)) | (2 << (6 * 4))),
            0,
        );
    }

    r
}

#[inline(always)]
pub(crate) unsafe fn xcbrtf_u1(d: VFloat) -> VFloat {
    let mut x: VFloat;
    let mut y: VFloat;
    let mut z: VFloat;
    let t: VFloat;
    let mut q2 = vcast_vf2_f_f(1.0, 0.0);
    let mut u: VFloat2;
    let mut v: VFloat2;
    let e: VInt2;
    let qu: VInt2;
    let re: VInt2;

    #[cfg(target_feature = "avx512f")]
    let s = d;

    e = vadd_vi2_vi2_vi2(vilogbk_vi2_vf(vabs_vf_vf(d)), vcast_vi2_i(1));
    let d = vldexp2_vf_vf_vi2(d, vneg_vi2_vi2(e));

    t = vadd_vf_vf_vf(vcast_vf_vi2(e), vcast_vf_f(6144.0));
    qu = vtruncate_vi2_vf(vmul_vf_vf_vf(t, vcast_vf_f(1.0 / 3.0)));
    re = vtruncate_vi2_vf(vsub_vf_vf_vf(
        t,
        vmul_vf_vf_vf(vcast_vf_vi2(qu), vcast_vf_f(3.0)),
    ));

    q2 = vsel_vf2_vo_vf2_vf2(
        veq_vo_vi2_vi2(re, vcast_vi2_i(1)),
        vcast_vf2_f_f(1.2599210739135742188f32, -2.4018701694217270415e-8f32),
        q2,
    );
    q2 = vsel_vf2_vo_vf2_vf2(
        veq_vo_vi2_vi2(re, vcast_vi2_i(2)),
        vcast_vf2_f_f(1.5874010324478149414f32, 1.9520385308169352356e-8f32),
        q2,
    );

    q2 = vf2setx_vf2_vf2_vf(q2, vmulsign_vf_vf_vf(vf2getx_vf_vf2(q2), d));
    q2 = vf2sety_vf2_vf2_vf(q2, vmulsign_vf_vf_vf(vf2gety_vf_vf2(q2), d));
    let d = vabs_vf_vf(d);

    x = vcast_vf_f(-0.601564466953277587890625f32);
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.8208892345428466796875f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-5.532182216644287109375f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(5.898262500762939453125f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-3.8095417022705078125f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.2241256237030029296875f32));

    y = vmul_vf_vf_vf(x, x);
    y = vmul_vf_vf_vf(y, y);
    x = vsub_vf_vf_vf(
        x,
        vmul_vf_vf_vf(vmlanp_vf_vf_vf_vf(d, y, x), vcast_vf_f(-1.0 / 3.0)),
    );

    z = x;

    u = dfmul_vf2_vf_vf(x, x);
    u = dfmul_vf2_vf2_vf2(u, u);
    u = dfmul_vf2_vf2_vf(u, d);
    u = dfadd2_vf2_vf2_vf(u, vneg_vf_vf(x));
    y = vadd_vf_vf_vf(vf2getx_vf_vf2(u), vf2gety_vf_vf2(u));

    y = vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(-2.0 / 3.0), y), z);
    v = dfadd2_vf2_vf2_vf(dfmul_vf2_vf_vf(z, z), y);
    v = dfmul_vf2_vf2_vf(v, d);
    v = dfmul_vf2_vf2_vf2(v, q2);
    z = vldexp2_vf_vf_vi2(
        vadd_vf_vf_vf(vf2getx_vf_vf2(v), vf2gety_vf_vf2(v)),
        vsub_vi2_vi2_vi2(qu, vcast_vi2_i(2048)),
    );

    z = vsel_vf_vo_vf_vf(
        visinf_vo_vf(d),
        vmulsign_vf_vf_vf(vcast_vf_f(f32::INFINITY), vf2getx_vf_vf2(q2)),
        z,
    );
    z = vsel_vf_vo_vf_vf(
        veq_vo_vf_vf(d, vcast_vf_f(0.0)),
        vreinterpret_vf_vm(vsignbit_vm_vf(vf2getx_vf_vf2(q2))),
        z,
    );

    #[cfg(target_feature = "avx512f")]
    {
        z = vsel_vf_vo_vf_vf(
            visinf_vo_vf(s),
            vmulsign_vf_vf_vf(vcast_vf_f(SLEEF_INFINITYf), s),
            z,
        );
        z = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(s, vcast_vf_f(0.0)),
            vmulsign_vf_vf_vf(vcast_vf_f(0.0), s),
            z,
        );
    }

    z
}

#[inline(always)]
pub(crate) unsafe fn xexpf(d: VFloat) -> VFloat {
    let q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(R_LN2_F)));

    let mut s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2_UF), d);
    s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2_LF), s);

    let mut u = vcast_vf_f(0.000198527617612853646278381f32);
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00139304355252534151077271f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833336077630519866943359f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416664853692054748535156f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166666671633720397949219f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5f32));

    u = vadd_vf_vf_vf(
        vcast_vf_f(1.0f32),
        vmla_vf_vf_vf_vf(vmul_vf_vf_vf(s, s), u, s),
    );

    u = vldexp2_vf_vf_vi2(u, q);

    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(
        vlt_vo_vf_vf(d, vcast_vf_f(-104.0)),
        vreinterpret_vm_vf(u),
    ));
    u = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(vcast_vf_f(100.0), d),
        vcast_vf_f(f32::INFINITY),
        u,
    );

    u
}

#[inline(always)]
unsafe fn expkf(d: VFloat2) -> VFloat {
    let u = vmul_vf_vf_vf(
        vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)),
        vcast_vf_f(R_LN2_F),
    );
    let q = vrint_vi2_vf(u);
    let mut s: VFloat2;
    let mut t: VFloat2;

    s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2_UF)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2_LF)));

    s = dfnormalize_vf2_vf2(s);

    let mut u = vcast_vf_f(0.00136324646882712841033936f32);
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.00836596917361021041870117f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.0416710823774337768554688f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.166665524244308471679688f32),
    );
    u = vmla_vf_vf_vf_vf(
        u,
        vf2getx_vf_vf2(s),
        vcast_vf_f(0.499999850988388061523438f32),
    );

    t = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfsqu_vf2_vf2(s), u));

    t = dfadd_vf2_vf_vf2(vcast_vf_f(1.0), t);
    let mut u = vadd_vf_vf_vf(vf2getx_vf_vf2(t), vf2gety_vf_vf2(t));
    u = vldexp_vf_vf_vi2(u, q);

    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(
        vlt_vo_vf_vf(vf2getx_vf_vf2(d), vcast_vf_f(-104.0)),
        vreinterpret_vm_vf(u),
    ));

    u
}

#[inline(always)]
unsafe fn logkf(d: VFloat) -> VFloat2 {
    let x: VFloat2;
    let x2: VFloat2;
    let mut t: VFloat;

    #[cfg(not(target_feature = "avx512f"))]
    let (m, e) = {
        let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
        let d = vsel_vf_vo_vf_vf(
            o,
            vmul_vf_vf_vf(d, vcast_vf_f(((1i64 << 32) as f32) * ((1i64 << 32) as f32))),
            d,
        );
        let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32 / 0.75f32)));
        let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
        e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
        (m, e)
    };

    #[cfg(target_feature = "avx512f")]
    let (m, e) = {
        let mut e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32 / 0.75f32)));
        e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f32), e);
        let m = vgetmant_vf_vf(d);
        (m, e)
    };

    x = dfdiv_vf2_vf2_vf2(
        dfadd2_vf2_vf_vf(vcast_vf_f(-1.0), m),
        dfadd2_vf2_vf_vf(vcast_vf_f(1.0), m),
    );
    x2 = dfsqu_vf2_vf2(x);

    t = vcast_vf_f(0.240320354700088500976562f32);
    t = vmla_vf_vf_vf_vf(
        t,
        vf2getx_vf_vf2(x2),
        vcast_vf_f(0.285112679004669189453125f32),
    );
    t = vmla_vf_vf_vf_vf(
        t,
        vf2getx_vf_vf2(x2),
        vcast_vf_f(0.400007992982864379882812f32),
    );
    let c = vcast_vf2_f_f(
        0.66666662693023681640625f32,
        3.69183861259614332084311e-9f32,
    );

    #[cfg(not(target_feature = "avx512f"))]
    let mut s = dfmul_vf2_vf2_vf(
        vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-9f32),
        vcast_vf_vi2(e),
    );

    #[cfg(target_feature = "avx512f")]
    let mut s = dfmul_vf2_vf2_vf(
        vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-9f32),
        e,
    );

    s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2.0)));
    s = dfadd_vf2_vf2_vf2(
        s,
        dfmul_vf2_vf2_vf2(
            dfmul_vf2_vf2_vf2(x2, x),
            dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(x2, t), c),
        ),
    );

    s
}

#[inline(always)]
pub(crate) unsafe fn xpowf(x: VFloat, y: VFloat) -> VFloat {
    let yisint = vor_vo_vo_vo(
        veq_vo_vf_vf(vtruncate_vf_vf(y), y),
        vgt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f((1 << 24) as f32)),
    );
    let yisodd = vand_vo_vo_vo(
        vand_vo_vo_vo(
            veq_vo_vi2_vi2(
                vand_vi2_vi2_vi2(vtruncate_vi2_vf(y), vcast_vi2_i(1)),
                vcast_vi2_i(1),
            ),
            yisint,
        ),
        vlt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f((1 << 24) as f32)),
    );

    let mut result = expkf(dfmul_vf2_vf2_vf(logkf(vabs_vf_vf(x)), y));

    result = vsel_vf_vo_vf_vf(visnan_vo_vf(result), vcast_vf_f(f32::INFINITY), result);

    result = vmul_vf_vf_vf(
        result,
        vsel_vf_vo_vf_vf(
            vgt_vo_vf_vf(x, vcast_vf_f(0.0)),
            vcast_vf_f(1.0),
            vsel_vf_vo_vf_vf(
                yisint,
                vsel_vf_vo_vf_vf(yisodd, vcast_vf_f(-1.0), vcast_vf_f(1.0)),
                vcast_vf_f(f32::NAN),
            ),
        ),
    );

    let efx = vmulsign_vf_vf_vf(vsub_vf_vf_vf(vabs_vf_vf(x), vcast_vf_f(1.0)), y);

    result = vsel_vf_vo_vf_vf(
        visinf_vo_vf(y),
        vreinterpret_vf_vm(vandnot_vm_vo32_vm(
            vlt_vo_vf_vf(efx, vcast_vf_f(0.0)),
            vreinterpret_vm_vf(vsel_vf_vo_vf_vf(
                veq_vo_vf_vf(efx, vcast_vf_f(0.0)),
                vcast_vf_f(1.0),
                vcast_vf_f(f32::INFINITY),
            )),
        )),
        result,
    );

    result = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0.0))),
        vmulsign_vf_vf_vf(
            vsel_vf_vo_vf_vf(
                vxor_vo_vo_vo(vsignbit_vo_vf(y), veq_vo_vf_vf(x, vcast_vf_f(0.0))),
                vcast_vf_f(0.0),
                vcast_vf_f(f32::INFINITY),
            ),
            vsel_vf_vo_vf_vf(yisodd, x, vcast_vf_f(1.0)),
        ),
        result,
    );

    result = vreinterpret_vf_vm(vor_vm_vo32_vm(
        vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)),
        vreinterpret_vm_vf(result),
    ));

    result = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            veq_vo_vf_vf(y, vcast_vf_f(0.0)),
            veq_vo_vf_vf(x, vcast_vf_f(1.0)),
        ),
        vcast_vf_f(1.0),
        result,
    );

    result
}

#[inline(always)]
pub(crate) unsafe fn xsinhf(x: VFloat) -> VFloat {
    let y = vabs_vf_vf(x);
    let mut d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0.0)));
    d = dfsub_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
    let mut y = vmul_vf_vf_vf(
        vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)),
        vcast_vf_f(0.5),
    );

    y = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89.0)),
            visnan_vo_vf(y),
        ),
        vcast_vf_f(f32::INFINITY),
        y,
    );
    y = vmulsign_vf_vf_vf(y, x);
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

    y
}

#[inline(always)]
pub(crate) unsafe fn xcoshf(x: VFloat) -> VFloat {
    let y = vabs_vf_vf(x);
    let mut d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0.0)));
    d = dfadd_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
    let mut y = vmul_vf_vf_vf(
        vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)),
        vcast_vf_f(0.5),
    );

    y = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89.0)),
            visnan_vo_vf(y),
        ),
        vcast_vf_f(f32::INFINITY),
        y,
    );
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

    y
}

#[inline(always)]
pub(crate) unsafe fn xtanhf(x: VFloat) -> VFloat {
    let y = vabs_vf_vf(x);
    let mut d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0.0)));
    let e = dfrec_vf2_vf2(d);
    d = dfdiv_vf2_vf2_vf2(
        dfadd_vf2_vf2_vf2(d, dfneg_vf2_vf2(e)),
        dfadd_vf2_vf2_vf2(d, e),
    );
    let mut y = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));

    y = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(8.664339742f32)),
            visnan_vo_vf(y),
        ),
        vcast_vf_f(1.0),
        y,
    );
    y = vmulsign_vf_vf_vf(y, x);
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

    y
}

#[inline(always)]
unsafe fn logk2f(d: VFloat2) -> VFloat2 {
    let x: VFloat2;
    let x2: VFloat2;
    let mut s: VFloat2;
    let mut t: VFloat;

    #[cfg(not(target_feature = "avx512f"))]
    let e = vilogbk_vi2_vf(vmul_vf_vf_vf(
        vf2getx_vf_vf2(d),
        vcast_vf_f(1.0f32 / 0.75f32),
    ));

    #[cfg(target_feature = "avx512f")]
    let e = vrint_vi2_vf(vgetexp_vf_vf(vmul_vf_vf_vf(
        vf2getx_vf_vf2(d),
        vcast_vf_f(1.0f32 / 0.75f32),
    )));

    let m = dfscale_vf2_vf2_vf(d, vpow2i_vf_vi2(vneg_vi2_vi2(e)));

    x = dfdiv_vf2_vf2_vf2(
        dfadd2_vf2_vf2_vf(m, vcast_vf_f(-1.0)),
        dfadd2_vf2_vf2_vf(m, vcast_vf_f(1.0)),
    );
    x2 = dfsqu_vf2_vf2(x);

    t = vcast_vf_f(0.2392828464508056640625f32);
    t = vmla_vf_vf_vf_vf(
        t,
        vf2getx_vf_vf2(x2),
        vcast_vf_f(0.28518211841583251953125f32),
    );
    t = vmla_vf_vf_vf_vf(
        t,
        vf2getx_vf_vf2(x2),
        vcast_vf_f(0.400005877017974853515625f32),
    );
    t = vmla_vf_vf_vf_vf(
        t,
        vf2getx_vf_vf2(x2),
        vcast_vf_f(0.666666686534881591796875f32),
    );

    s = dfmul_vf2_vf2_vf(
        vcast_vf2_vf_vf(
            vcast_vf_f(0.69314718246459960938f32),
            vcast_vf_f(-1.904654323148236017e-9f32),
        ),
        vcast_vf_vi2(e),
    );
    s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2.0)));
    s = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfmul_vf2_vf2_vf2(x2, x), t));

    s
}

#[inline(always)]
pub(crate) unsafe fn xasinhf(x: VFloat) -> VFloat {
    let y = vabs_vf_vf(x);
    let o = vgt_vo_vf_vf(y, vcast_vf_f(1.0));
    let mut d: VFloat2;

    d = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf(x), vcast_vf2_vf_vf(y, vcast_vf_f(0.0)));
    d = dfsqrt_vf2_vf2(dfadd2_vf2_vf2_vf(dfsqu_vf2_vf2(d), vcast_vf_f(1.0)));
    d = vsel_vf2_vo_vf2_vf2(o, dfmul_vf2_vf2_vf(d, y), d);

    d = logk2f(dfnormalize_vf2_vf2(dfadd2_vf2_vf2_vf(d, x)));
    let mut y = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));

    y = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(SQRT_FLT_MAX as f32)),
            visnan_vo_vf(y),
        ),
        vmulsign_vf_vf_vf(vcast_vf_f((1e+300 * 1e+300) as f32), x),
        y,
    );
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));
    y = vsel_vf_vo_vf_vf(visnegzero_vo_vf(x), vcast_vf_f(-0.0), y);

    y
}

#[inline(always)]
unsafe fn expk2f(d: VFloat2) -> VFloat2 {
    let u = vmul_vf_vf_vf(
        vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)),
        vcast_vf_f(R_LN2_F),
    );
    let q = vrint_vi2_vf(u);
    let mut s: VFloat2;
    let mut t: VFloat2;

    s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2_UF)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2_LF)));

    let mut u = vcast_vf_f(0.1980960224e-3f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.1394256484e-2f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.8333456703e-2f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.4166637361e-1f32));

    t = dfadd2_vf2_vf2_vf(
        dfmul_vf2_vf2_vf(s, u),
        vcast_vf_f(0.166666659414234244790680580464f32),
    );
    t = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf2(s, t), vcast_vf_f(0.5));
    t = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf2(dfsqu_vf2_vf2(s), t));

    t = dfadd_vf2_vf_vf2(vcast_vf_f(1.0), t);

    t = vf2setx_vf2_vf2_vf(t, vldexp2_vf_vf_vi2(vf2getx_vf_vf2(t), q));
    t = vf2sety_vf2_vf2_vf(t, vldexp2_vf_vf_vi2(vf2gety_vf_vf2(t), q));

    t = vf2setx_vf2_vf2_vf(
        t,
        vreinterpret_vf_vm(vandnot_vm_vo32_vm(
            vlt_vo_vf_vf(vf2getx_vf_vf2(d), vcast_vf_f(-104.0)),
            vreinterpret_vm_vf(vf2getx_vf_vf2(t)),
        )),
    );
    t = vf2sety_vf2_vf2_vf(
        t,
        vreinterpret_vf_vm(vandnot_vm_vo32_vm(
            vlt_vo_vf_vf(vf2getx_vf_vf2(d), vcast_vf_f(-104.0)),
            vreinterpret_vm_vf(vf2gety_vf_vf2(t)),
        )),
    );

    t
}

#[inline(always)]
pub(crate) unsafe fn xatanhf(x: VFloat) -> VFloat {
    let y = vabs_vf_vf(x);
    let d = logk2f(dfdiv_vf2_vf2_vf2(
        dfadd2_vf2_vf_vf(vcast_vf_f(1.0), y),
        dfadd2_vf2_vf_vf(vcast_vf_f(1.0), vneg_vf_vf(y)),
    ));
    let mut y = vreinterpret_vf_vm(vor_vm_vo32_vm(
        vgt_vo_vf_vf(y, vcast_vf_f(1.0)),
        vreinterpret_vm_vf(vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(y, vcast_vf_f(1.0)),
            vcast_vf_f(f32::INFINITY),
            vmul_vf_vf_vf(
                vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)),
                vcast_vf_f(0.5),
            ),
        )),
    ));

    y = vreinterpret_vf_vm(vor_vm_vo32_vm(
        vor_vo_vo_vo(visinf_vo_vf(x), visnan_vo_vf(y)),
        vreinterpret_vm_vf(y),
    ));
    y = vmulsign_vf_vf_vf(y, x);
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

    y
}

#[inline(always)]
pub(crate) unsafe fn xacoshf(x: VFloat) -> VFloat {
    let d = logk2f(dfadd2_vf2_vf2_vf(
        dfmul_vf2_vf2_vf2(
            dfsqrt_vf2_vf2(dfadd2_vf2_vf_vf(x, vcast_vf_f(1.0))),
            dfsqrt_vf2_vf2(dfadd2_vf2_vf_vf(x, vcast_vf_f(-1.0))),
        ),
        x,
    ));
    let mut y = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));

    y = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(SQRT_FLT_MAX as f32)),
            visnan_vo_vf(y),
        ),
        vcast_vf_f(f32::INFINITY),
        y,
    );

    y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(
        veq_vo_vf_vf(x, vcast_vf_f(1.0)),
        vreinterpret_vm_vf(y),
    ));

    y = vreinterpret_vf_vm(vor_vm_vo32_vm(
        vlt_vo_vf_vf(x, vcast_vf_f(1.0)),
        vreinterpret_vm_vf(y),
    ));
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

    y
}

#[inline(always)]
pub(crate) unsafe fn xexp2f(d: VFloat) -> VFloat {
    let u = vrint_vf_vf(d);
    let q = vrint_vi2_vf(u);

    let s = vsub_vf_vf_vf(d, u);

    let mut u = vcast_vf_f(0.1535920892e-3f32);
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.1339262701e-2f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.9618384764e-2f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5550347269e-1f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.2402264476f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.6931471825f32));

    #[cfg(target_feature = "fma")]
    {
        u = vfma_vf_vf_vf_vf(u, s, vcast_vf_f(1.0));
    }

    #[cfg(not(target_feature = "fma"))]
    {
        u = vf2getx_vf_vf2(dfnormalize_vf2_vf2(dfadd_vf2_vf_vf2(
            vcast_vf_f(1.0),
            dfmul_vf2_vf_vf(u, s),
        )));
    }

    u = vldexp2_vf_vf_vi2(u, q);

    u = vsel_vf_vo_vf_vf(
        vge_vo_vf_vf(d, vcast_vf_f(128.0)),
        vcast_vf_f(f32::INFINITY),
        u,
    );
    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(
        vlt_vo_vf_vf(d, vcast_vf_f(-150.0)),
        vreinterpret_vm_vf(u),
    ));

    u
}

#[inline(always)]
pub(crate) unsafe fn xexp10f(d: VFloat) -> VFloat {
    let u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(LOG10_2 as f32)));
    let q = vrint_vi2_vf(u);

    let mut s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-L10_UF), d);
    s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-L10_LF), s);

    let mut u = vcast_vf_f(0.6802555919e-1f32);
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.2078080326f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5393903852f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.1171245337e1f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.2034678698e1f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.2650949001e1f32));

    let x = dfadd_vf2_vf2_vf(
        vcast_vf2_f_f(2.3025851249694824219f32, -3.1705172516493593157e-8f32),
        vmul_vf_vf_vf(u, s),
    );
    u = vf2getx_vf_vf2(dfnormalize_vf2_vf2(dfadd_vf2_vf_vf2(
        vcast_vf_f(1.0),
        dfmul_vf2_vf2_vf(x, s),
    )));

    u = vldexp2_vf_vf_vi2(u, q);

    u = vsel_vf_vo_vf_vf(
        vgt_vo_vf_vf(d, vcast_vf_f(38.5318394191036238941387f32)),
        vcast_vf_f(f32::INFINITY),
        u,
    );
    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(
        vlt_vo_vf_vf(d, vcast_vf_f(-50.0)),
        vreinterpret_vm_vf(u),
    ));

    u
}

#[inline(always)]
pub(crate) unsafe fn xexpm1f(a: VFloat) -> VFloat {
    let d = dfadd2_vf2_vf2_vf(
        expk2f(vcast_vf2_vf_vf(a, vcast_vf_f(0.0))),
        vcast_vf_f(-1.0),
    );
    let mut x = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));

    x = vsel_vf_vo_vf_vf(
        vgt_vo_vf_vf(a, vcast_vf_f(88.72283172607421875f32)),
        vcast_vf_f(f32::INFINITY),
        x,
    );
    x = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(a, vcast_vf_f(-16.63553233343868742601357f32)),
        vcast_vf_f(-1.0),
        x,
    );
    x = vsel_vf_vo_vf_vf(visnegzero_vo_vf(a), vcast_vf_f(-0.0), x);

    x
}

#[inline(always)]
pub(crate) unsafe fn xlog10f(d: VFloat) -> VFloat {
    let x: VFloat2;
    let mut t: VFloat;
    let x2: VFloat;

    #[cfg(not(target_feature = "avx512f"))]
    let (m, e) = {
        let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
        let d = vsel_vf_vo_vf_vf(
            o,
            vmul_vf_vf_vf(
                d,
                vcast_vf_f(((1u64 << 32u64) as f32) * ((1u64 << 32u64) as f32)),
            ),
            d,
        );
        let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0 / 0.75)));
        let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
        e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
        (m, e)
    };

    #[cfg(target_feature = "avx512f")]
    let (m, e) = {
        let mut e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0 / 0.75)));
        e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0), e);
        let m = vgetmant_vf_vf(d);
        (m, e)
    };

    x = dfdiv_vf2_vf2_vf2(
        dfadd2_vf2_vf_vf(vcast_vf_f(-1.0), m),
        dfadd2_vf2_vf_vf(vcast_vf_f(1.0), m),
    );
    x2 = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));

    t = vcast_vf_f(0.1314289868f32);
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.1735493541f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.2895309627f32));

    #[cfg(not(target_feature = "avx512f"))]
    let mut s = dfmul_vf2_vf2_vf(
        vcast_vf2_f_f(0.30103001f32, -1.432098889e-8f32),
        vcast_vf_vi2(e),
    );

    #[cfg(target_feature = "avx512f")]
    let mut s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.30103001f32, -1.432098889e-8f32), e);

    s = dfadd_vf2_vf2_vf2(
        s,
        dfmul_vf2_vf2_vf2(x, vcast_vf2_f_f(0.868588984f32, -2.170757285e-8f32)),
    );
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)), t));

    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(s), vf2gety_vf_vf2(s));

    #[cfg(not(target_feature = "avx512f"))]
    {
        r = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(f32::INFINITY), r);
        r = vsel_vf_vo_vf_vf(
            vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0.0)), visnan_vo_vf(d)),
            vcast_vf_f(f32::NAN),
            r,
        );
        r = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(d, vcast_vf_f(0.0)),
            vcast_vf_f(f32::NEG_INFINITY),
            r,
        );
    }

    #[cfg(target_feature = "avx512f")]
    {
        r = vfixup_vf_vf_vf_vi2_i(
            r,
            d,
            vcast_vi2_i((4 << (2 * 4)) | (3 << (4 * 4)) | (5 << (5 * 4)) | (2 << (6 * 4))),
            0,
        );
    }

    r
}

#[inline(always)]
pub(crate) unsafe fn xlog2f(d: VFloat) -> VFloat {
    let x: VFloat2;
    let mut t: VFloat;
    let x2: VFloat;

    #[cfg(not(target_feature = "avx512f"))]
    let (m, e) = {
        let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
        let d = vsel_vf_vo_vf_vf(
            o,
            vmul_vf_vf_vf(
                d,
                vcast_vf_f(((1u64 << 32u64) as f32) * ((1u64 << 32u64) as f32)),
            ),
            d,
        );
        let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0 / 0.75)));
        let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
        e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
        (m, e)
    };

    #[cfg(target_feature = "avx512f")]
    let (m, e) = {
        let mut e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0 / 0.75)));
        e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0), e);
        let m = vgetmant_vf_vf(d);
        (m, e)
    };

    x = dfdiv_vf2_vf2_vf2(
        dfadd2_vf2_vf_vf(vcast_vf_f(-1.0), m),
        dfadd2_vf2_vf_vf(vcast_vf_f(1.0), m),
    );
    x2 = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));

    t = vcast_vf_f(0.4374550283f32);
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.5764790177f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.961801290512f32));

    #[cfg(not(target_feature = "avx512f"))]
    let mut s = dfadd2_vf2_vf_vf2(
        vcast_vf_vi2(e),
        dfmul_vf2_vf2_vf2(
            x,
            vcast_vf2_f_f(2.8853900432586669922f32, 3.2734474483568488616e-8f32),
        ),
    );

    #[cfg(target_feature = "avx512f")]
    let mut s = dfadd2_vf2_vf_vf2(
        e,
        dfmul_vf2_vf2_vf2(
            x,
            vcast_vf2_f_f(2.8853900432586669922f32, 3.2734474483568488616e-8f32),
        ),
    );

    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)), t));

    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(s), vf2gety_vf_vf2(s));

    #[cfg(not(target_feature = "avx512f"))]
    {
        r = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(f32::INFINITY), r);
        r = vsel_vf_vo_vf_vf(
            vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0.0)), visnan_vo_vf(d)),
            vcast_vf_f(f32::NAN),
            r,
        );
        r = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(d, vcast_vf_f(0.0)),
            vcast_vf_f(f32::NEG_INFINITY),
            r,
        );
    }

    #[cfg(target_feature = "avx512f")]
    {
        r = vfixup_vf_vf_vf_vi2_i(
            r,
            d,
            vcast_vi2_i((4 << (2 * 4)) | (3 << (4 * 4)) | (5 << (5 * 4)) | (2 << (6 * 4))),
            0,
        );
    }

    r
}

#[inline(always)]
pub(crate) unsafe fn xlog1pf(d: VFloat) -> VFloat {
    let x: VFloat2;
    let mut t: VFloat;
    let x2: VFloat;

    let dp1 = vadd_vf_vf_vf(d, vcast_vf_f(1.0));

    #[cfg(not(target_feature = "avx512f"))]
    let (m, mut s) = {
        let o = vlt_vo_vf_vf(dp1, vcast_vf_f(SLEEF_FLT_MIN));
        let dp1 = vsel_vf_vo_vf_vf(
            o,
            vmul_vf_vf_vf(
                dp1,
                vcast_vf_f(((1u64 << 32u64) as f32) * ((1u64 << 32u64) as f32)),
            ),
            dp1,
        );
        let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(dp1, vcast_vf_f(1.0 / 0.75)));
        t = vldexp3_vf_vf_vi2(vcast_vf_f(1.0), vneg_vi2_vi2(e));
        let m = vmla_vf_vf_vf_vf(d, t, vsub_vf_vf_vf(t, vcast_vf_f(1.0)));
        e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
        let s = dfmul_vf2_vf2_vf(
            vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-9f32),
            vcast_vf_vi2(e),
        );
        (m, s)
    };

    #[cfg(target_feature = "avx512f")]
    let (m, mut s) = {
        let mut e = vgetexp_vf_vf(vmul_vf_vf_vf(dp1, vcast_vf_f(1.0 / 0.75)));
        e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0), e);
        t = vldexp3_vf_vf_vi2(vcast_vf_f(1.0), vneg_vi2_vi2(vrint_vi2_vf(e)));
        let m = vmla_vf_vf_vf_vf(d, t, vsub_vf_vf_vf(t, vcast_vf_f(1.0)));
        let s = dfmul_vf2_vf2_vf(
            vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-9f32),
            e,
        );
        (m, s)
    };

    x = dfdiv_vf2_vf2_vf2(
        vcast_vf2_vf_vf(m, vcast_vf_f(0.0)),
        dfadd_vf2_vf_vf(vcast_vf_f(2.0), m),
    );
    x2 = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));

    t = vcast_vf_f(0.3027294874f32);
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.3996108174f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.666669488f32));

    s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2.0)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)), t));

    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(s), vf2gety_vf_vf2(s));

    let ocore = vle_vo_vf_vf(d, vcast_vf_f(LOG1PF_BOUND));
    if !(vtestallones_i_vo32(ocore) != 0) {
        r = vsel_vf_vo_vf_vf(ocore, r, xlogf_u1(d));
    }

    r = vreinterpret_vf_vm(vor_vm_vo32_vm(
        vgt_vo_vf_vf(vcast_vf_f(-1.0), d),
        vreinterpret_vm_vf(r),
    ));
    r = vsel_vf_vo_vf_vf(
        veq_vo_vf_vf(d, vcast_vf_f(-1.0)),
        vcast_vf_f(f32::NEG_INFINITY),
        r,
    );
    r = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0), r);

    r
}

#[inline(always)]
pub(crate) unsafe fn xsqrtf_u05(d: VFloat) -> VFloat {
    #[cfg(target_feature = "fma")]
    {
        use helper::vfmanp_vf_vf_vf_vf;
        use helper::vfmapn_vf_vf_vf_vf;
        let q: VFloat;
        let mut w: VFloat;
        let mut x: VFloat;
        let mut y: VFloat;
        let mut z: VFloat;

        let d = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(d, vcast_vf_f(0.0)), vcast_vf_f(f32::NAN), d);

        let o = vlt_vo_vf_vf(d, vcast_vf_f(5.293955920339377e-23f32));
        let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(1.888946593147858e22f32)), d);
        q = vsel_vf_vo_vf_vf(o, vcast_vf_f(7.275957614183426e-12f32), vcast_vf_f(1.0));

        y = vreinterpret_vf_vi2(vsub_vi2_vi2_vi2(
            vcast_vi2_i(0x5f3759df),
            vsrl_vi2_vi2_i::<1>(vreinterpret_vi2_vf(d)),
        ));

        x = vmul_vf_vf_vf(d, y);
        w = vmul_vf_vf_vf(vcast_vf_f(0.5), y);
        y = vfmanp_vf_vf_vf_vf(x, w, vcast_vf_f(0.5));
        x = vfma_vf_vf_vf_vf(x, y, x);
        w = vfma_vf_vf_vf_vf(w, y, w);
        y = vfmanp_vf_vf_vf_vf(x, w, vcast_vf_f(0.5));
        x = vfma_vf_vf_vf_vf(x, y, x);
        w = vfma_vf_vf_vf_vf(w, y, w);

        y = vfmanp_vf_vf_vf_vf(x, w, vcast_vf_f(1.5));
        w = vadd_vf_vf_vf(w, w);
        w = vmul_vf_vf_vf(w, y);
        x = vmul_vf_vf_vf(w, d);
        y = vfmapn_vf_vf_vf_vf(w, d, x);
        z = vfmanp_vf_vf_vf_vf(w, x, vcast_vf_f(1.0));

        z = vfmanp_vf_vf_vf_vf(w, y, z);
        w = vmul_vf_vf_vf(vcast_vf_f(0.5), x);
        w = vfma_vf_vf_vf_vf(w, z, y);
        w = vadd_vf_vf_vf(w, x);

        w = vmul_vf_vf_vf(w, q);

        w = vsel_vf_vo_vf_vf(
            vor_vo_vo_vo(
                veq_vo_vf_vf(d, vcast_vf_f(0.0)),
                veq_vo_vf_vf(d, vcast_vf_f(f32::INFINITY)),
            ),
            d,
            w,
        );

        w = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(d, vcast_vf_f(0.0)), vcast_vf_f(f32::NAN), w);

        w
    }

    #[cfg(not(target_feature = "fma"))]
    {
        let mut q: VFloat;
        let mut o: Vopmask;

        let d = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(d, vcast_vf_f(0.0)), vcast_vf_f(f32::NAN), d);

        o = vlt_vo_vf_vf(d, vcast_vf_f(5.293955920339377e-23f32));
        let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(1.888946593147858e22f32)), d);
        q = vsel_vf_vo_vf_vf(
            o,
            vcast_vf_f(7.275957614183426e-12f32 * 0.5),
            vcast_vf_f(0.5),
        );

        o = vgt_vo_vf_vf(d, vcast_vf_f(1.8446744073709552e19f32));
        let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(5.421010862427522e-20f32)), d);
        q = vsel_vf_vo_vf_vf(o, vcast_vf_f(4294967296.0 * 0.5), q);

        let mut x = vreinterpret_vf_vi2(vsub_vi2_vi2_vi2(
            vcast_vi2_i(0x5f375a86),
            vsrl_vi2_vi2_i::<1>(vreinterpret_vi2_vf(vadd_vf_vf_vf(d, vcast_vf_f(1e-45f32)))),
        ));

        x = vmul_vf_vf_vf(
            x,
            vsub_vf_vf_vf(
                vcast_vf_f(1.5),
                vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(0.5), d), x), x),
            ),
        );
        x = vmul_vf_vf_vf(
            x,
            vsub_vf_vf_vf(
                vcast_vf_f(1.5),
                vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(0.5), d), x), x),
            ),
        );
        x = vmul_vf_vf_vf(
            x,
            vsub_vf_vf_vf(
                vcast_vf_f(1.5),
                vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(0.5), d), x), x),
            ),
        );
        x = vmul_vf_vf_vf(x, d);

        let d2 = dfmul_vf2_vf2_vf2(dfadd2_vf2_vf_vf2(d, dfmul_vf2_vf_vf(x, x)), dfrec_vf2_vf(x));

        x = vmul_vf_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d2), vf2gety_vf_vf2(d2)), q);

        x = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(f32::INFINITY), x);
        x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0.0)), d, x);

        x
    }
}

#[inline(always)]
pub(crate) unsafe fn xhypotf_u05(x: VFloat, y: VFloat) -> VFloat {
    let x = vabs_vf_vf(x);
    let y = vabs_vf_vf(y);
    let min = vmin_vf_vf_vf(x, y);
    let mut n = min;
    let max = vmax_vf_vf_vf(x, y);
    let mut d = max;

    let o = vlt_vo_vf_vf(max, vcast_vf_f(SLEEF_FLT_MIN));
    n = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(n, vcast_vf_f(f32::from_bits(1 << 24))), n);
    d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(f32::from_bits(1 << 24))), d);

    let t = dfdiv_vf2_vf2_vf2(
        vcast_vf2_vf_vf(n, vcast_vf_f(0.0)),
        vcast_vf2_vf_vf(d, vcast_vf_f(0.0)),
    );
    let t = dfmul_vf2_vf2_vf(
        dfsqrt_vf2_vf2(dfadd2_vf2_vf2_vf(dfsqu_vf2_vf2(t), vcast_vf_f(1.0))),
        max,
    );

    let mut ret = vadd_vf_vf_vf(vf2getx_vf_vf2(t), vf2gety_vf_vf2(t));

    ret = vsel_vf_vo_vf_vf(visnan_vo_vf(ret), vcast_vf_f(f32::INFINITY), ret);
    ret = vsel_vf_vo_vf_vf(veq_vo_vf_vf(min, vcast_vf_f(0.0)), max, ret);
    ret = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)),
        vcast_vf_f(f32::NAN),
        ret,
    );
    ret = vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            veq_vo_vf_vf(x, vcast_vf_f(f32::INFINITY)),
            veq_vo_vf_vf(y, vcast_vf_f(f32::INFINITY)),
        ),
        vcast_vf_f(f32::INFINITY),
        ret,
    );

    ret
}

#[inline(always)]
pub(crate) unsafe fn xtruncf(x: VFloat) -> VFloat {
    let fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
    vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            visinf_vo_vf(x),
            vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f((1i64 << 23) as f32)),
        ),
        x,
        vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x),
    )
}

#[inline(always)]
pub(crate) unsafe fn xfloorf(x: VFloat) -> VFloat {
    let mut fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
    fr = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(fr, vcast_vf_f(0.0)),
        vadd_vf_vf_vf(fr, vcast_vf_f(1.0)),
        fr,
    );
    vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            visinf_vo_vf(x),
            vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f((1i64 << 23) as f32)),
        ),
        x,
        vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x),
    )
}

#[inline(always)]
pub(crate) unsafe fn xceilf(x: VFloat) -> VFloat {
    let mut fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
    fr = vsel_vf_vo_vf_vf(
        vle_vo_vf_vf(fr, vcast_vf_f(0.0)),
        fr,
        vsub_vf_vf_vf(fr, vcast_vf_f(1.0)),
    );
    vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            visinf_vo_vf(x),
            vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f((1i64 << 23) as f32)),
        ),
        x,
        vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x),
    )
}

#[inline(always)]
pub(crate) unsafe fn xroundf(d: VFloat) -> VFloat {
    let mut x = vadd_vf_vf_vf(d, vcast_vf_f(0.5));
    let mut fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
    x = vsel_vf_vo_vf_vf(
        vand_vo_vo_vo(
            vle_vo_vf_vf(x, vcast_vf_f(0.0)),
            veq_vo_vf_vf(fr, vcast_vf_f(0.0)),
        ),
        vsub_vf_vf_vf(x, vcast_vf_f(1.0)),
        x,
    );
    fr = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(fr, vcast_vf_f(0.0)),
        vadd_vf_vf_vf(fr, vcast_vf_f(1.0)),
        fr,
    );
    x = vsel_vf_vo_vf_vf(
        veq_vo_vf_vf(d, vcast_vf_f(0.4999999701976776123)),
        vcast_vf_f(0.0),
        x,
    );
    vsel_vf_vo_vf_vf(
        vor_vo_vo_vo(
            visinf_vo_vf(d),
            vge_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f((1i64 << 23) as f32)),
        ),
        d,
        vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), d),
    )
}

#[inline(always)]
pub(crate) unsafe fn xcopysignf(x: VFloat, y: VFloat) -> VFloat {
    vcopysign_vf_vf_vf(x, y)
}

#[inline(always)]
unsafe fn dfmla_vf2_vf_vf2_vf2(x: VFloat, y: VFloat2, z: VFloat2) -> VFloat2 {
    dfadd_vf2_vf2_vf2(z, dfmul_vf2_vf2_vf(y, x))
}
#[inline(always)]
unsafe fn poly2df_b(x: VFloat, c1: VFloat2, c0: VFloat2) -> VFloat2 {
    dfmla_vf2_vf_vf2_vf2(x, c1, c0)
}

#[inline(always)]
unsafe fn poly2df(x: VFloat, c1: VFloat, c0: VFloat2) -> VFloat2 {
    dfmla_vf2_vf_vf2_vf2(x, vcast_vf2_vf_vf(c1, vcast_vf_f(0.0)), c0)
}
#[inline(always)]
unsafe fn poly4df(x: VFloat, c3: VFloat, c2: VFloat2, c1: VFloat2, c0: VFloat2) -> VFloat2 {
    dfmla_vf2_vf_vf2_vf2(
        vmul_vf_vf_vf(x, x),
        poly2df(x, c3, c2),
        poly2df_b(x, c1, c0),
    )
}

#[inline(always)]
pub(crate) unsafe fn xerff_u1(a: VFloat) -> VFloat {
    let t: VFloat;
    let x = vabs_vf_vf(a);
    let mut t2: VFloat2;
    let x2 = vmul_vf_vf_vf(x, x);
    let x4 = vmul_vf_vf_vf(x2, x2);
    let o25 = vle_vo_vf_vf(x, vcast_vf_f(2.5));

    if vtestallones_i_vo32(o25) != 0 {
        t = poly6(
            x,
            x2,
            x4,
            -0.4360447008e-6,
            0.6867515367e-5,
            -0.30451567e-4,
            0.9808536561e-4,
            0.2395523916e-3,
            0.1459901541e-3,
        );
        t2 = poly4df(
            x,
            t,
            vcast_vf2_f_f(0.0092883445322513580322, -2.7863745897025330755e-11),
            vcast_vf2_f_f(0.042275499552488327026, 1.3461399289988106057e-9),
            vcast_vf2_f_f(0.070523701608180999756, -3.6616309318707365163e-9),
        );
        t2 = dfadd_vf2_vf_vf2(vcast_vf_f(1.0), dfmul_vf2_vf2_vf(t2, x));
        t2 = dfsqu_vf2_vf2(t2);
        t2 = dfsqu_vf2_vf2(t2);
        t2 = dfsqu_vf2_vf2(t2);
        t2 = dfsqu_vf2_vf2(t2);
        t2 = dfrec_vf2_vf2(t2);
    } else {
        t = poly6_(
            x,
            x2,
            x4,
            vsel_vf_vo_f_f(o25, -0.4360447008e-6, -0.1130012848e-6),
            vsel_vf_vo_f_f(o25, 0.6867515367e-5, 0.4115272986e-5),
            vsel_vf_vo_f_f(o25, -0.30451567e-4, -0.6928304356e-4),
            vsel_vf_vo_f_f(o25, 0.9808536561e-4, 0.7172692567e-3),
            vsel_vf_vo_f_f(o25, 0.2395523916e-3, -0.5131045356e-2),
            vsel_vf_vo_f_f(o25, 0.1459901541e-3, 0.2708637156e-1),
        );
        t2 = poly4df(
            x,
            t,
            vsel_vf2_vo_vf2_vf2(
                o25,
                vcast_vf2_f_f(0.0092883445322513580322, -2.7863745897025330755e-11),
                vcast_vf2_f_f(-0.11064319312572479248, 3.7050452777225283007e-9),
            ),
            vsel_vf2_vo_vf2_vf2(
                o25,
                vcast_vf2_f_f(0.042275499552488327026, 1.3461399289988106057e-9),
                vcast_vf2_f_f(-0.63192230463027954102, -2.0200432585073177859e-8),
            ),
            vsel_vf2_vo_vf2_vf2(
                o25,
                vcast_vf2_f_f(0.070523701608180999756, -3.6616309318707365163e-9),
                vcast_vf2_f_f(-1.1296638250350952148, 2.5515120196453259252e-8),
            ),
        );
        t2 = dfmul_vf2_vf2_vf(t2, x);
        let s2 = dfadd_vf2_vf_vf2(vcast_vf_f(1.0), t2);
        let s2 = dfsqu_vf2_vf2(s2);
        let s2 = dfsqu_vf2_vf2(s2);
        let s2 = dfsqu_vf2_vf2(s2);
        let s2 = dfsqu_vf2_vf2(s2);
        let s2 = dfrec_vf2_vf2(s2);
        t2 = vsel_vf2_vo_vf2_vf2(o25, s2, vcast_vf2_vf_vf(expkf(t2), vcast_vf_f(0.0)));
    }

    t2 = dfadd2_vf2_vf2_vf(t2, vcast_vf_f(-1.0));
    t2 = vsel_vf2_vo_vf2_vf2(
        vlt_vo_vf_vf(x, vcast_vf_f(1e-4)),
        dfmul_vf2_vf2_vf(
            vcast_vf2_f_f(-1.1283792257308959961, 5.8635383422197591097e-8),
            x,
        ),
        t2,
    );

    let mut z = vneg_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(t2), vf2gety_vf_vf2(t2)));
    z = vsel_vf_vo_vf_vf(vge_vo_vf_vf(x, vcast_vf_f(6.0)), vcast_vf_f(1.0), z);
    z = vsel_vf_vo_vf_vf(visinf_vo_vf(a), vcast_vf_f(1.0), z);
    z = vsel_vf_vo_vf_vf(veq_vo_vf_vf(a, vcast_vf_f(0.0)), vcast_vf_f(0.0), z);
    z = vmulsign_vf_vf_vf(z, a);

    z
}

#[inline(always)]
pub(crate) unsafe fn xmaxf(x: VFloat, y: VFloat) -> VFloat {
    vsel_vf_vo_vf_vf(visnan_vo_vf(y), x, vmax_vf_vf_vf(x, y))
}

#[inline(always)]
pub(crate) unsafe fn xminf(x: VFloat, y: VFloat) -> VFloat {
    vsel_vf_vo_vf_vf(
        visnan_vo_vf(y),
        x,
        vsel_vf_vo_vf_vf(vgt_vo_vf_vf(y, x), x, y),
    )
}
