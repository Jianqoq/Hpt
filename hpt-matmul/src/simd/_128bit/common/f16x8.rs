
/// a vector of 8 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct f16x8(pub(crate) [half::f16; 8]);

impl f16x8 {
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const half::f16) -> Self {
        let mut result = [half::f16::ZERO; 8];
        for i in 0..8 {
            result[i] = unsafe { *ptr.add(i) };
        }
        f16x8(result)
    }
}