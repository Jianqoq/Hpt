use tensor_common::slice::Slice;
use tensor_traits::{ CommonBounds, ShapeManipulate, TensorCreator };

use crate::{ slice::SliceOps, tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T> where T: CommonBounds {
    pub fn one_hot<U>(
        axis: i64,
        indices: &Tensor<U>,
        depth: i64,
        on_val: T,
        off_val: T
    ) -> anyhow::Result<_Tensor<T>> {
        let mut res_shape = indices.layout.shape().inner().clone();
        let axis = if axis < 0 { axis + (res_shape.len() as i64) } else { axis };
        res_shape.insert(axis as usize, depth);
        let ret = _Tensor::<T>::empty(&res_shape)?;
        let swpaed = ret.swap_axes(0, axis)?;
        println!("sliced: {:?}", swpaed);
        Ok(swpaed)
    }
}
