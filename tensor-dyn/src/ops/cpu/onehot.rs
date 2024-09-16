use tensor_common::err_handler::ErrHandler;
use tensor_traits::{CommonBounds, ShapeManipulate, TensorCreator, TensorInfo};

use crate::tensor_base::_Tensor;

impl<T> _Tensor<T>
where
    T: CommonBounds,
{
    pub fn onehot(&self, depth: usize, mut axis: i64, _true_val: T, false_val: T) -> anyhow::Result<Self> {
        let mut new_shape = self.shape().inner().clone();
        ErrHandler::check_index_in_range_mut(self.ndim(), &mut axis)?;
        new_shape.insert(axis as usize, depth as i64);
        let res = _Tensor::<T>::full(false_val, new_shape)?;
        let mut permute_axes = (0..res.ndim()).collect::<Vec<usize>>();
        permute_axes.retain(|x| *x != axis as usize);
        permute_axes.push(axis as usize);
        let permuted_res = res.permute(permute_axes)?;
        assert_eq!(&permuted_res.shape()[..res.ndim() - 1], self.shape().inner().as_slice());
        todo!()
    }
}
