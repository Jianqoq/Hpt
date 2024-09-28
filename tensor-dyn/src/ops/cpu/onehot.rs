use tensor_common::err_handler::ErrHandler;
use tensor_traits::{ CommonBounds, ShapeManipulate, TensorCreator, TensorInfo };

use crate::tensor_base::_Tensor;

impl<T> _Tensor<T> where T: CommonBounds {
    /// Converts the input tensor into a one-hot encoded tensor along a specified axis.
    ///
    /// This method transforms the input tensor into a one-hot encoded format, where the values
    /// along the specified axis are converted into vectors of size `depth`. Each vector contains
    /// a `true_val` at the index specified by the input tensor and `false_val` elsewhere.
    ///
    /// # Arguments
    ///
    /// * `depth` - The size of the one-hot vectors. This represents the number of unique categories
    ///   for the one-hot encoding. Each element in the input tensor will be transformed into a one-hot
    ///   vector of this length.
    /// * `axis` - The axis along which the one-hot encoding is applied. If the axis is negative, it is
    ///   treated as counting from the last dimension of the tensor. The new one-hot vectors will be inserted
    ///   along this axis.
    /// * `_true_val` - The value that will be placed at the position corresponding to the one-hot index (usually 1).
    /// * `false_val` - The value that will fill the other positions in the one-hot vector (usually 0).
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the one-hot encoded values.
    pub fn onehot(
        &self,
        depth: usize,
        mut axis: i64,
        _true_val: T,
        false_val: T
    ) -> anyhow::Result<Self> {
        let mut new_shape = self.shape().inner().clone();
        ErrHandler::check_index_in_range_mut(self.ndim(), &mut axis)?;
        new_shape.insert(axis as usize, depth as i64);
        let res = _Tensor::<T>::full(false_val, new_shape)?;
        let mut permute_axes = (0..res.ndim()).collect::<Vec<usize>>();
        permute_axes.retain(|x| *x != (axis as usize));
        permute_axes.push(axis as usize);
        let permuted_res = res.permute(permute_axes)?;
        assert_eq!(&permuted_res.shape()[..res.ndim() - 1], self.shape().inner().as_slice());
        todo!()
    }
}
