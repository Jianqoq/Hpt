pub trait IntoVec<T> {
    fn into_vec(self) -> T;
}

#[cfg(target_feature = "avx2")]
mod into_vec {

    use super::IntoVec;
    use crate::convertion::VecConvertor;
    use crate::vectors::_256bit::*;
    use tensor_macros::impl_into_vec;
    impl_into_vec!();
}

#[cfg(all(
    any(target_feature = "sse2", target_feature = "neon"),
    not(target_feature = "avx2")
))]
mod into_vec {
    use super::IntoVec;
    use crate::convertion::VecConvertor;
    use crate::vectors::_128bit::*;
    use tensor_macros::impl_into_vec;
    impl_into_vec!();
}
