#![allow(unused)]

use crate::{arch_simd::sleef::arch::helper::{vcast_vf_f, vmla_vf_vf_vf_vf}, VFloat};

#[inline(always)]
pub(crate) unsafe fn poly2(x: VFloat, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x, vcast_vf_f(c1), vcast_vf_f(c0))
}

#[inline(always)]
pub(crate) unsafe fn poly3(x: VFloat, x2: VFloat, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x2, vcast_vf_f(c2), vmla_vf_vf_vf_vf(x, vcast_vf_f(c1), vcast_vf_f(c0)))
}

#[inline(always)]
pub(crate) unsafe fn poly4(x: VFloat, x2: VFloat, c3: f32, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(
        x2,
        vmla_vf_vf_vf_vf(x, vcast_vf_f(c3), vcast_vf_f(c2)),
        vmla_vf_vf_vf_vf(x, vcast_vf_f(c1), vcast_vf_f(c0)),
    )
}

#[inline(always)]
pub(crate) unsafe fn poly5(x: VFloat, x2: VFloat, x4: VFloat, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x4, vcast_vf_f(c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly6(x: VFloat, x2: VFloat, x4: VFloat, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x4, poly2(x, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly7(x: VFloat, x2: VFloat, x4: VFloat, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x4, poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly8(x: VFloat, x2: VFloat, x4: VFloat, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x4, poly4(x, x2, c7, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly9(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(x8, vcast_vf_f(c8), poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly10(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(x8, poly2(x, c9, c8), poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly11(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    ca: f32, c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(x8, poly3(x, x2, ca, c9, c8), poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly12(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    cb: f32, ca: f32, c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x8,
        poly4(x, x2, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly13(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    cc: f32, cb: f32, ca: f32, c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x8,
        poly5(x, x2, x4, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly14(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x8,
        poly6(x, x2, x4, cd, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly15(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x8,
        poly7(x, x2, x4, ce, cd, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly16(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat,
    cf: f32, ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32, c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x8,
        poly8(x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly17(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat, x16: VFloat,
    d0: f32, cf: f32, ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32,
    c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(x16, vcast_vf_f(d0), 
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly18(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat, x16: VFloat,
    d1: f32, d0: f32, cf: f32, ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32,
    c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(x16, poly2(x, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly19(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat, x16: VFloat,
    d2: f32, d1: f32, d0: f32, cf: f32, ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32,
    c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x16,
        poly3(x, x2, d2, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly20(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat, x16: VFloat,
    d3: f32, d2: f32, d1: f32, d0: f32, cf: f32, ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32,
    c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x16,
        poly4(x, x2, d3, d2, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}

#[inline(always)]
pub(crate) unsafe fn poly21(
    x: VFloat, x2: VFloat, x4: VFloat, x8: VFloat, x16: VFloat,
    d4: f32, d3: f32, d2: f32, d1: f32, d0: f32, cf: f32, ce: f32, cd: f32, cc: f32, cb: f32, ca: f32, c9: f32, c8: f32,
    c7: f32, c6: f32, c5: f32, c4: f32, c3: f32, c2: f32, c1: f32, c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(
        x16,
        poly5(x, x2, x4, d4, d3, d2, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0)
    )
}


#[inline(always)]
pub(crate) unsafe fn poly2_(x: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x, c1, c0)
}

#[inline(always)]
pub(crate) unsafe fn poly3_(x: VFloat, x2: VFloat, c2: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x2, c2, vmla_vf_vf_vf_vf(x, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly4_(x: VFloat, x2: VFloat, c3: VFloat, c2: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(
        x2,
        vmla_vf_vf_vf_vf(x, c3, c2),
        vmla_vf_vf_vf_vf(x, c1, c0),
    )
}

#[inline(always)]
pub(crate) unsafe fn poly5_(x: VFloat, x2: VFloat, x4: VFloat, c4: VFloat, c3: VFloat, c2: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x4, c4, poly4_(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly6_(x: VFloat, x2: VFloat, x4: VFloat, c5: VFloat, c4: VFloat, c3: VFloat, c2: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x4, poly2_(x, c5, c4), poly4_(x, x2, c3, c2, c1, c0))
}