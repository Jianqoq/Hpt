#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::arch_simd::sleef::arch::helper_avx2 as helper;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "avx2")
))]
use crate::arch_simd::sleef::arch::helper_sse as helper;
use crate::sleef_types::{VDouble, VMask, Vopmask};
use helper::{
    vabs_vd_vd, vadd64_vm_vm_vm, vadd_vd_vd_vd, vadd_vi_vi_vi, vand_vm_vm_vm, vand_vo_vo_vo,
    vandnot_vm_vm_vm, vcast_vd_d, vcast_vd_vi, vcast_vi_i, vcastu_vm_vi, veq64_vo_vm_vm,
    veq_vo_vd_vd, vge_vo_vd_vd, vgt_vo_vd_vd, visinf_vo_vd, vle_vo_vd_vd, vlt_vo_vd_vd,
    vmla_vd_vd_vd_vd, vmul_vd_vd_vd, vneq_vo_vd_vd, vor_vm_vm_vm, vor_vo_vo_vo, vreinterpret_vd_vm,
    vreinterpret_vm_vd, vsel_vd_vo_vd_vd, vsll_vi_vi_i, vsra_vi_vi_i, vsub_vd_vd_vd, vsub_vi_vi_vi,
    vtruncate_vi_vd, vxor_vm_vm_vm,
};

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct DI {
    d: VDouble,     i: VInt,    }

#[inline(always)]
pub(crate) unsafe fn digetd_vd_di(d: DI) -> VDouble {
        d.d
}

#[inline(always)]
pub(crate) unsafe fn digeti_vi_di(d: DI) -> VInt {
        d.i
}

#[inline(always)]
pub(crate) unsafe fn disetdi_di_vd_vi(d: VDouble, i: VInt) -> DI {
        DI { d, i }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct DDI {
    dd: VDouble2,
    i: VInt,
}

#[inline(always)]
pub(crate) unsafe fn ddigetdd_vd2_ddi(d: DDI) -> VDouble2 {
        d.dd
}

#[inline(always)]
pub(crate) unsafe fn ddigeti_vi_ddi(d: DDI) -> VInt {
        d.i
}

#[inline(always)]
pub(crate) unsafe fn ddisetddi_ddi_vd2_vi(v: VDouble2, i: VInt) -> DDI {
        DDI { dd: v, i }
}

#[inline(always)]
pub(crate) unsafe fn ddisetdd_ddi_ddi_vd2(mut ddi: DDI, v: VDouble2) -> DDI {
        ddi.dd = v;
    ddi
}

#[inline(always)]
pub(crate) unsafe fn visnegzero_vo_vd(d: VDouble) -> Vopmask {
        veq64_vo_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)))
}

#[inline(always)]
pub(crate) unsafe fn vsignbit_vm_vd(d: VDouble) -> VMask {
        vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)))
}

#[inline(always)]
pub(crate) unsafe fn vsignbit_vo_vd(d: VDouble) -> Vopmask {
    veq64_vo_vm_vm(
        vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))),
        vreinterpret_vm_vd(vcast_vd_d(-0.0)),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmulsign_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
        vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)))
}

#[inline(always)]
pub(crate) unsafe fn vorsign_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
        vreinterpret_vd_vm(vor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)))
}

#[inline(always)]
pub(crate) unsafe fn vcopysign_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
        vreinterpret_vd_vm(vxor_vm_vm_vm(
        vandnot_vm_vm_vm(vreinterpret_vm_vd(vcast_vd_d(-0.0)), vreinterpret_vm_vd(x)),
        vand_vm_vm_vm(vreinterpret_vm_vd(vcast_vd_d(-0.0)), vreinterpret_vm_vd(y)),
    ))
}

#[inline(always)]
pub(crate) unsafe fn vtruncate2_vd_vd(x: VDouble) -> VDouble {
    let mut fr = vsub_vd_vd_vd(
        x,
        vmul_vd_vd_vd(
            vcast_vd_d((1i64 << 31) as f64),
            vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(
                x,
                vcast_vd_d(1.0 / (1i64 << 31) as f64),
            ))),
        ),
    );

    fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));

    vsel_vd_vo_vd_vd(
        vor_vo_vo_vo(
            visinf_vo_vd(x),
            vge_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d((1i64 << 52) as f64)),
        ),
        x,
        vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x),
    )
}

#[inline(always)]
pub(crate) unsafe fn vround2_vd_vd(d: VDouble) -> VDouble {
        let mut x = vadd_vd_vd_vd(d, vcast_vd_d(0.5));
    let mut fr = vsub_vd_vd_vd(
        x,
        vmul_vd_vd_vd(
            vcast_vd_d((1i64 << 31) as f64),
            vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(
                x,
                vcast_vd_d(1.0 / (1i64 << 31) as f64),
            ))),
        ),
    );

    fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));

        x = vsel_vd_vo_vd_vd(
        vand_vo_vo_vo(
            vle_vo_vd_vd(x, vcast_vd_d(0.0)),
            veq_vo_vd_vd(fr, vcast_vd_d(0.0)),
        ),
        vsub_vd_vd_vd(x, vcast_vd_d(1.0)),
        x,
    );

    fr = vsel_vd_vo_vd_vd(
        vlt_vo_vd_vd(fr, vcast_vd_d(0.0)),
        vadd_vd_vd_vd(fr, vcast_vd_d(1.0)),
        fr,
    );

        x = vsel_vd_vo_vd_vd(
        veq_vo_vd_vd(d, vcast_vd_d(0.49999999999999994449)),
        vcast_vd_d(0.0),
        x,
    );

        vsel_vd_vo_vd_vd(
        vor_vo_vo_vo(
            visinf_vo_vd(d),
            vge_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d((1i64 << 52) as f64)),
        ),
        d,
        vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), d),
    )
}

#[inline(always)]
pub(crate) unsafe fn vrint2_vd_vd(d: VDouble) -> VDouble {
    let c = vmulsign_vd_vd_vd(vcast_vd_d((1i64 << 52) as f64), d);
    vsel_vd_vo_vd_vd(
        vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d((1i64 << 52) as f64)),
        d,
        vorsign_vd_vd_vd(vsub_vd_vd_vd(vadd_vd_vd_vd(d, c), c), d),
    )
}

#[inline(always)]
pub(crate) unsafe fn visint_vo_vd(d: VDouble) -> Vopmask {
        veq_vo_vd_vd(vrint2_vd_vd(d), d)
}

#[inline(always)]
pub(crate) unsafe fn visodd_vo_vd(d: VDouble) -> Vopmask {
        let x = vmul_vd_vd_vd(d, vcast_vd_d(0.5));
    vneq_vo_vd_vd(vrint2_vd_vd(x), x)
}

#[cfg(not(target_feature = "avx512f"))]
use crate::sleef_types::VInt;

use super::dd::VDouble2;
#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
pub(crate) unsafe fn vilogbk_vi_vd(d: VDouble) -> VInt {
    
    use helper::{
        vand_vi_vi_vi, vcast_vi_i, vcast_vo32_vo64, vcastu_vi_vm, vsel_vi_vo_vi_vi, vsrl_vi_vi_i,
        vsub_vi_vi_vi,
    };
    let o = vlt_vo_vd_vd(d, vcast_vd_d(4.9090934652977266E-91));
    let d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(vcast_vd_d(2.037035976334486E90), d), d);
    let mut q = vcastu_vi_vm(vreinterpret_vm_vd(d));
    q = vand_vi_vi_vi(q, vcast_vi_i((((1u32 << 12) - 1) << 20) as i32));
    q = vsrl_vi_vi_i::<20>(q);
    q = vsub_vi_vi_vi(
        q,
        vsel_vi_vo_vi_vi(
            vcast_vo32_vo64(o),
            vcast_vi_i(300 + 0x3ff),
            vcast_vi_i(0x3ff),
        ),
    );
    q
}

#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
pub(crate) unsafe fn vilogb2k_vi_vd(d: VDouble) -> VInt {
    
    use helper::{
        vand_vi_vi_vi, vcast_vi_i, vcastu_vi_vm, vsrl_vi_vi_i, vsub_vi_vi_vi,
    };
    let mut q = vcastu_vi_vm(vreinterpret_vm_vd(d));
    q = vsrl_vi_vi_i::<20>(q);
    q = vand_vi_vi_vi(q, vcast_vi_i(0x7ff));
    q = vsub_vi_vi_vi(q, vcast_vi_i(0x3ff));
    q
}

#[inline(always)]
pub(crate) unsafe fn vpow2i_vd_vi(q: VInt) -> VDouble {
        let q = vadd_vi_vi_vi(vcast_vi_i(0x3ff), q);
    let r = vcastu_vm_vi(vsll_vi_vi_i::<20>(q));
    vreinterpret_vd_vm(r)
}

#[inline(always)]
pub(crate) unsafe fn vldexp2_vd_vd_vi(d: VDouble, e: VInt) -> VDouble {
        vmul_vd_vd_vd(
        vmul_vd_vd_vd(d, vpow2i_vd_vi(vsra_vi_vi_i::<1>(e))),
        vpow2i_vd_vi(vsub_vi_vi_vi(e, vsra_vi_vi_i::<1>(e))),
    )
}

#[inline(always)]
pub(crate) unsafe fn vldexp3_vd_vd_vi(d: VDouble, q: VInt) -> VDouble {
        vreinterpret_vd_vm(vadd64_vm_vm_vm(
        vreinterpret_vm_vd(d),
        vcastu_vm_vi(vsll_vi_vi_i::<20>(q)),
    ))
}

#[inline(always)]
pub(crate) unsafe fn rempisub(x: VDouble) -> DI {
    let c = vmulsign_vd_vd_vd(vcast_vd_d((1i64 << 52) as f64), x);
    let rint4x = vsel_vd_vo_vd_vd(
        vgt_vo_vd_vd(
            vabs_vd_vd(vmul_vd_vd_vd(vcast_vd_d(4.0), x)),
            vcast_vd_d((1i64 << 52) as f64),
        ),
        vmul_vd_vd_vd(vcast_vd_d(4.0), x),
        vorsign_vd_vd_vd(vsub_vd_vd_vd(vmla_vd_vd_vd_vd(vcast_vd_d(4.0), x, c), c), x),
    );
    let rintx = vsel_vd_vo_vd_vd(
        vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d((1i64 << 52) as f64)),
        x,
        vorsign_vd_vd_vd(vsub_vd_vd_vd(vadd_vd_vd_vd(x, c), c), x),
    );
    disetdi_di_vd_vi(
        vmla_vd_vd_vd_vd(vcast_vd_d(-0.25), rint4x, x),
        vtruncate_vi_vd(vmla_vd_vd_vd_vd(vcast_vd_d(-4.0), rintx, rint4x)),
    )
}
