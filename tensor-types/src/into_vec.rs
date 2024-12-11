/// a trait to convert a vector to another vector
pub trait IntoVec<T> {
    /// convert a vector to another vector T
    fn into_vec(self) -> T;
}

#[cfg(target_feature = "avx2")]
mod into_vec {
    use super::IntoVec;
    use crate::convertion::VecConvertor;
    #[cfg(feature = "stdsimd")]
    use crate::vectors::std_simd::_256bit::*;
    use tensor_macros::impl_into_vec;
    impl_into_vec!();
}

#[cfg(
    all(
        any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
        not(target_feature = "avx2")
    )
)]
mod into_vec {
    use super::IntoVec;
    use crate::convertion::VecConvertor;
    #[cfg(feature = "stdsimd")]
    use crate::vectors::std_simd::_128bit::*;
    use tensor_macros::impl_into_vec;
    impl_into_vec!();
}
