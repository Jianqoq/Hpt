/// a trait to convert a vector to another vector
pub trait IntoVec<T> {
    /// convert a vector to another vector T
    fn into_vec(self) -> T;
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod into_vec {
    use super::IntoVec;
    use crate::convertion::VecConvertor;
    use crate::simd::_256bit::common::*;
    use hpt_macros::impl_into_vec;
    impl_into_vec!();
}

#[cfg(target_feature = "avx512f")]
mod into_vec {
    use super::IntoVec;
    use crate::convertion::VecConvertor;
    use crate::simd::_512bit::common::*;
    use hpt_macros::impl_into_vec;
    impl_into_vec!();
}

#[cfg(all(
    any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
    not(target_feature = "avx2")
))]
mod into_vec {
    use super::IntoVec;
    use crate::convertion::VecConvertor;
    use crate::simd::_128bit::common::*;
    use hpt_macros::impl_into_vec;
    impl_into_vec!();
}
