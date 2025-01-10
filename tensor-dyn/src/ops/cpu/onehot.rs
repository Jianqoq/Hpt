use tensor_common::{
    error::{base::TensorError, shape::ShapeError},
    shape::shape_utils::mt_intervals,
};
use tensor_traits::{CommonBounds, ShapeManipulate, TensorCreator, TensorInfo};

use crate::{tensor_base::_Tensor, Tensor, THREAD_POOL};

impl<T> _Tensor<T>
where
    T: CommonBounds,
{
    pub fn onehot(
        &self,
        depth: usize,
        mut axis: i64,
        _true_val: T,
        false_val: T,
    ) -> std::result::Result<Self, TensorError> {
        let mut new_shape = self.shape().inner().clone();
        if axis < 0 {
            axis += self.ndim() as i64;
        }
        ShapeError::check_index_out_of_range(axis, self.ndim() as i64)?;
        axis += 1;
        new_shape.insert(axis as usize, depth as i64);
        let res = _Tensor::<T>::full(false_val, new_shape)?;
        let mut permute_axes = (0..res.ndim()).collect::<Vec<usize>>();
        permute_axes.retain(|x| *x != (axis as usize));
        permute_axes.push(axis as usize);
        let permuted_res = res.permute(permute_axes)?;
        assert_eq!(
            &permuted_res.shape()[..res.ndim() - 1],
            self.shape().inner().as_slice()
        );
        let inner_loop_size = *permuted_res.shape().last().unwrap();
        let outer_loop_size = permuted_res.size() as i64 / inner_loop_size;

        let last_strides = *permuted_res.strides().last().unwrap();

        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if outer_loop_size < (pool.max_count() as i64) {
                outer_loop_size as usize
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(outer_loop_size as usize, num_threads);
            for (start, end) in intervals.into_iter() {
                let mut res_ptr = permuted_res.ptr();
                let inp_shape = self.shape().clone();
                let inp_strides = self.strides().clone();
                let permuted_res_shape = permuted_res.shape().clone();
                let permuted_res_strides = permuted_res.strides().clone();
                let inp_ptr = self.ptr();
                pool.execute(move || {
                    for i in start..end {
                        let mut amount = i;
                        let mut res_amount = i as i64 * inner_loop_size;
                        let mut index = 0i64;
                        let mut res_index = 0i64;
                        for j in (0..inp_shape.len()).rev() {
                            index += ((amount as i64) % inp_shape[j]) * inp_strides[j];
                            amount /= inp_shape[j] as usize;
                        }
                        for j in (0..permuted_res_shape.len()).rev() {
                            res_index += ((res_amount as i64) % permuted_res_shape[j]) * permuted_res_strides[j];
                            res_amount /= permuted_res_shape[j];
                        }
                        let dp = inp_ptr[index].to_i64();
                        res_ptr[res_index + dp * last_strides] = _true_val;
                    }
                });
            }
            pool.join();
        });

        Ok(res)
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds,
{
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
        axis: i64,
        _true_val: T,
        false_val: T,
    ) -> std::result::Result<Tensor<T>, TensorError> {
        Ok(self.inner.onehot(depth, axis, _true_val, false_val)?.into())
    }
}
