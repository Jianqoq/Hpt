#![allow(unused)]
use hpt_types::dtype::TypeCommon;

#[inline(always)]
pub(crate) fn prefetch_b<T: TypeCommon>(b: *const <T as TypeCommon>::Vec, offset: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(
            b.add(offset) as *const i8,
            std::arch::x86_64::_MM_HINT_T0,
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]", 
            in(reg) b.add(offset),
            options(nostack, preserves_flags)
        );
    }
}

#[inline(always)]
pub(crate) fn prefetch_a<T: TypeCommon>(a: *const T, offset: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(
            a.add(offset) as *const i8,
            std::arch::x86_64::_MM_HINT_T0,
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]", 
            in(reg) a.add(offset),
            options(nostack, preserves_flags)
        );
    }
}