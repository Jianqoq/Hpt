use tensor_traits::CommonBounds;
use std::hash::Hash;
use crate::tensor_base::_Tensor;

impl<T> _Tensor<T> where T: Hash + Eq + CommonBounds {
    pub fn unique(&self) -> anyhow::Result<_Tensor<T>> {
        todo!()
    }
}
