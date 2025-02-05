#![allow(unused)]
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

use helper::{vcast_vd_d, vcast_vf_f, vmla_vd_vd_vd_vd, vmla_vf_vf_vf_vf};

use crate::sleef_types::{VDouble, VFloat};

#[inline(always)]
pub(crate) unsafe fn poly2(x: VFloat, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(x, vcast_vf_f(c1), vcast_vf_f(c0))
}

// #[inline(always)]
// pub(crate) unsafe fn poly3(x: VFloat, x2: VFloat, c2: f32, c1: f32, c0: f32) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x2,
//         vcast_vf_f(c2),
//         vmla_vf_vf_vf_vf(x, vcast_vf_f(c1), vcast_vf_f(c0)),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly4(x: VFloat, x2: VFloat, c3: f32, c2: f32, c1: f32, c0: f32) -> VFloat {
    vmla_vf_vf_vf_vf(
        x2,
        vmla_vf_vf_vf_vf(x, vcast_vf_f(c3), vcast_vf_f(c2)),
        vmla_vf_vf_vf_vf(x, vcast_vf_f(c1), vcast_vf_f(c0)),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly5(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(x4, vcast_vf_f(c4), poly4(x, x2, c3, c2, c1, c0))
// }

#[inline(always)]
pub(crate) unsafe fn poly6(
    x: VFloat,
    x2: VFloat,
    x4: VFloat,
    c5: f32,
    c4: f32,
    c3: f32,
    c2: f32,
    c1: f32,
    c0: f32,
) -> VFloat {
    vmla_vf_vf_vf_vf(x4, poly2(x, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

// #[inline(always)]
// pub(crate) unsafe fn poly7(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(x4, poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
// }

// #[inline(always)]
// pub(crate) unsafe fn poly8(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x4,
//         poly4(x, x2, c7, c6, c5, c4),
//         poly4(x, x2, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly9(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         vcast_vf_f(c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly10(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly2(x, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly11(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly3(x, x2, ca, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly12(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly4(x, x2, cb, ca, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly13(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly5(x, x2, x4, cc, cb, ca, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly14(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly6(x, x2, x4, cd, cc, cb, ca, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly15(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly7(x, x2, x4, ce, cd, cc, cb, ca, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly16(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     cf: f32,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x8,
//         poly8(x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
//         poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly17(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     x16: VFloat,
//     d0: f32,
//     cf: f32,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x16,
//         vcast_vf_f(d0),
//         poly16(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly18(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     x16: VFloat,
//     d1: f32,
//     d0: f32,
//     cf: f32,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x16,
//         poly2(x, d1, d0),
//         poly16(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly19(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     x16: VFloat,
//     d2: f32,
//     d1: f32,
//     d0: f32,
//     cf: f32,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x16,
//         poly3(x, x2, d2, d1, d0),
//         poly16(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly20(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     x16: VFloat,
//     d3: f32,
//     d2: f32,
//     d1: f32,
//     d0: f32,
//     cf: f32,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x16,
//         poly4(x, x2, d3, d2, d1, d0),
//         poly16(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly21(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     x8: VFloat,
//     x16: VFloat,
//     d4: f32,
//     d3: f32,
//     d2: f32,
//     d1: f32,
//     d0: f32,
//     cf: f32,
//     ce: f32,
//     cd: f32,
//     cc: f32,
//     cb: f32,
//     ca: f32,
//     c9: f32,
//     c8: f32,
//     c7: f32,
//     c6: f32,
//     c5: f32,
//     c4: f32,
//     c3: f32,
//     c2: f32,
//     c1: f32,
//     c0: f32,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(
//         x16,
//         poly5(x, x2, x4, d4, d3, d2, d1, d0),
//         poly16(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly2_(x: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x, c1, c0)
}

#[inline(always)]
pub(crate) unsafe fn poly3_(x: VFloat, x2: VFloat, c2: VFloat, c1: VFloat, c0: VFloat) -> VFloat {
    vmla_vf_vf_vf_vf(x2, c2, vmla_vf_vf_vf_vf(x, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly4_(
    x: VFloat,
    x2: VFloat,
    c3: VFloat,
    c2: VFloat,
    c1: VFloat,
    c0: VFloat,
) -> VFloat {
    vmla_vf_vf_vf_vf(x2, vmla_vf_vf_vf_vf(x, c3, c2), vmla_vf_vf_vf_vf(x, c1, c0))
}

// #[inline(always)]
// pub(crate) unsafe fn poly5_(
//     x: VFloat,
//     x2: VFloat,
//     x4: VFloat,
//     c4: VFloat,
//     c3: VFloat,
//     c2: VFloat,
//     c1: VFloat,
//     c0: VFloat,
// ) -> VFloat {
//     vmla_vf_vf_vf_vf(x4, c4, poly4_(x, x2, c3, c2, c1, c0))
// }

#[inline(always)]
pub(crate) unsafe fn poly6_(
    x: VFloat,
    x2: VFloat,
    x4: VFloat,
    c5: VFloat,
    c4: VFloat,
    c3: VFloat,
    c2: VFloat,
    c1: VFloat,
    c0: VFloat,
) -> VFloat {
    vmla_vf_vf_vf_vf(x4, poly2_(x, c5, c4), poly4_(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly2d(x: VDouble, c1: f64, c0: f64) -> VDouble {
    vmla_vd_vd_vd_vd(x, vcast_vd_d(c1), vcast_vd_d(c0))
}

#[inline(always)]
pub(crate) unsafe fn poly3d(x: VDouble, x2: VDouble, c2: f64, c1: f64, c0: f64) -> VDouble {
    vmla_vd_vd_vd_vd(
        x2,
        vcast_vd_d(c2),
        vmla_vd_vd_vd_vd(x, vcast_vd_d(c1), vcast_vd_d(c0)),
    )
}

#[inline(always)]
pub(crate) unsafe fn poly4d(
    x: VDouble,
    x2: VDouble,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x2,
        vmla_vd_vd_vd_vd(x, vcast_vd_d(c3), vcast_vd_d(c2)),
        vmla_vd_vd_vd_vd(x, vcast_vd_d(c1), vcast_vd_d(c0)),
    )
}

#[inline(always)]
pub(crate) unsafe fn poly5d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(x4, vcast_vd_d(c4), poly4d(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly6d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(x4, poly2d(x, c5, c4), poly4d(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly7d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(x4, poly3d(x, x2, c6, c5, c4), poly4d(x, x2, c3, c2, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly8d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x4,
        poly4d(x, x2, c7, c6, c5, c4),
        poly4d(x, x2, c3, c2, c1, c0),
    )
}

#[inline(always)]
pub(crate) unsafe fn poly9d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x8,
        vcast_vd_d(c8),
        poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

#[inline(always)]
pub(crate) unsafe fn poly10d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x8,
        poly2d(x, c9, c8),
        poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly11d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly3d(x, x2, ca, c9, c8),
//         poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly12d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x8,
        poly4d(x, x2, cb, ca, c9, c8),
        poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly13d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly5d(x, x2, x4, cc, cb, ca, c9, c8),
//         poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly14d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     cd: f64,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly6d(x, x2, x4, cd, cc, cb, ca, c9, c8),
//         poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly15d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     ce: f64,
//     cd: f64,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly7d(x, x2, x4, ce, cd, cc, cb, ca, c9, c8),
//         poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly16d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x8,
        poly8d(x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
        poly8d(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly17d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d0: f64,
//     cf: f64,
//     ce: f64,
//     cd: f64,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         vcast_vd_d(d0),
//         poly16d(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly18d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d1: f64,
//     d0: f64,
//     cf: f64,
//     ce: f64,
//     cd: f64,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         poly2d(x, d1, d0),
//         poly16d(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly19d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d2: f64,
//     d1: f64,
//     d0: f64,
//     cf: f64,
//     ce: f64,
//     cd: f64,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         poly3d(x, x2, d2, d1, d0),
//         poly16d(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly20d(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d3: f64,
//     d2: f64,
//     d1: f64,
//     d0: f64,
//     cf: f64,
//     ce: f64,
//     cd: f64,
//     cc: f64,
//     cb: f64,
//     ca: f64,
//     c9: f64,
//     c8: f64,
//     c7: f64,
//     c6: f64,
//     c5: f64,
//     c4: f64,
//     c3: f64,
//     c2: f64,
//     c1: f64,
//     c0: f64,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         poly4d(x, x2, d3, d2, d1, d0),
//         poly16d(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly21d(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    x16: VDouble,
    d4: f64,
    d3: f64,
    d2: f64,
    d1: f64,
    d0: f64,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x16,
        poly5d(x, x2, x4, d4, d3, d2, d1, d0),
        poly16d(
            x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly2d_(x: VDouble, c1: VDouble, c0: VDouble) -> VDouble {
//     vmla_vd_vd_vd_vd(x, c1, c0)
// }

// #[inline(always)]
// pub(crate) unsafe fn poly3d_(
//     x: VDouble,
//     x2: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(x2, c2, vmla_vd_vd_vd_vd(x, c1, c0))
// }

#[inline(always)]
pub(crate) unsafe fn poly4d_(
    x: VDouble,
    x2: VDouble,
    c3: VDouble,
    c2: VDouble,
    c1: VDouble,
    c0: VDouble,
) -> VDouble {
    vmla_vd_vd_vd_vd(x2, vmla_vd_vd_vd_vd(x, c3, c2), vmla_vd_vd_vd_vd(x, c1, c0))
}

#[inline(always)]
pub(crate) unsafe fn poly5d_(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    c4: VDouble,
    c3: VDouble,
    c2: VDouble,
    c1: VDouble,
    c0: VDouble,
) -> VDouble {
    vmla_vd_vd_vd_vd(x4, c4, poly4d_(x, x2, c3, c2, c1, c0))
}

// #[inline(always)]
// pub(crate) unsafe fn poly6d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(x4, poly2d_(x, c5, c4), poly4d_(x, x2, c3, c2, c1, c0))
// }

// #[inline(always)]
// pub(crate) unsafe fn poly7d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x4,
//         poly3d_(x, x2, c6, c5, c4),
//         poly4d_(x, x2, c3, c2, c1, c0),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly8d_(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    c7: VDouble,
    c6: VDouble,
    c5: VDouble,
    c4: VDouble,
    c3: VDouble,
    c2: VDouble,
    c1: VDouble,
    c0: VDouble,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x4,
        poly4d_(x, x2, c7, c6, c5, c4),
        poly4d_(x, x2, c3, c2, c1, c0),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly9d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(x8, c8, poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
// }

// #[inline(always)]
// pub(crate) unsafe fn poly10d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly2d_(x, c9, c8),
//         poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly11d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly3d_(x, x2, ca, c9, c8),
//         poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly12d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly4d_(x, x2, cb, ca, c9, c8),
//         poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly13d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly5d_(x, x2, x4, cc, cb, ca, c9, c8),
//         poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly14d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     cd: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly6d_(x, x2, x4, cd, cc, cb, ca, c9, c8),
//         poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly15d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     ce: VDouble,
//     cd: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x8,
//         poly7d_(x, x2, x4, ce, cd, cc, cb, ca, c9, c8),
//         poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly16d_(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    cf: VDouble,
    ce: VDouble,
    cd: VDouble,
    cc: VDouble,
    cb: VDouble,
    ca: VDouble,
    c9: VDouble,
    c8: VDouble,
    c7: VDouble,
    c6: VDouble,
    c5: VDouble,
    c4: VDouble,
    c3: VDouble,
    c2: VDouble,
    c1: VDouble,
    c0: VDouble,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x8,
        poly8d_(x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
        poly8d_(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

// #[inline(always)]
// pub(crate) unsafe fn poly17d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d0: VDouble,
//     cf: VDouble,
//     ce: VDouble,
//     cd: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         d0,
//         poly16d_(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly18d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d1: VDouble,
//     d0: VDouble,
//     cf: VDouble,
//     ce: VDouble,
//     cd: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         poly2d_(x, d1, d0),
//         poly16d_(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly19d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d2: VDouble,
//     d1: VDouble,
//     d0: VDouble,
//     cf: VDouble,
//     ce: VDouble,
//     cd: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         poly3d_(x, x2, d2, d1, d0),
//         poly16d_(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

// #[inline(always)]
// pub(crate) unsafe fn poly20d_(
//     x: VDouble,
//     x2: VDouble,
//     x4: VDouble,
//     x8: VDouble,
//     x16: VDouble,
//     d3: VDouble,
//     d2: VDouble,
//     d1: VDouble,
//     d0: VDouble,
//     cf: VDouble,
//     ce: VDouble,
//     cd: VDouble,
//     cc: VDouble,
//     cb: VDouble,
//     ca: VDouble,
//     c9: VDouble,
//     c8: VDouble,
//     c7: VDouble,
//     c6: VDouble,
//     c5: VDouble,
//     c4: VDouble,
//     c3: VDouble,
//     c2: VDouble,
//     c1: VDouble,
//     c0: VDouble,
// ) -> VDouble {
//     vmla_vd_vd_vd_vd(
//         x16,
//         poly4d_(x, x2, d3, d2, d1, d0),
//         poly16d_(
//             x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
//         ),
//     )
// }

#[inline(always)]
pub(crate) unsafe fn poly21d_(
    x: VDouble,
    x2: VDouble,
    x4: VDouble,
    x8: VDouble,
    x16: VDouble,
    d4: VDouble,
    d3: VDouble,
    d2: VDouble,
    d1: VDouble,
    d0: VDouble,
    cf: VDouble,
    ce: VDouble,
    cd: VDouble,
    cc: VDouble,
    cb: VDouble,
    ca: VDouble,
    c9: VDouble,
    c8: VDouble,
    c7: VDouble,
    c6: VDouble,
    c5: VDouble,
    c4: VDouble,
    c3: VDouble,
    c2: VDouble,
    c1: VDouble,
    c0: VDouble,
) -> VDouble {
    vmla_vd_vd_vd_vd(
        x16,
        poly5d_(x, x2, x4, d4, d3, d2, d1, d0),
        poly16d_(
            x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}
