use tensor_common::shape::shape_utils::mt_intervals;
use tensor_traits::{CommonBounds, ShapeManipulate, TensorCreator, TensorInfo};

use crate::{backend::Cpu, tensor_base::_Tensor, Tensor, THREAD_POOL};

impl<T> _Tensor<T, Cpu>
where
    T: CommonBounds,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn gather(&self, indices: &_Tensor<i64, Cpu>, axis: i64) -> anyhow::Result<Self> {
        let axis = if axis < 0 {
            self.ndim() as i64 + axis
        } else {
            axis
        } as usize;

        let mut permute_axes = (0..indices.ndim()).collect::<Vec<usize>>();
        permute_axes.retain(|x| *x != (axis as usize));
        permute_axes.push(axis as usize);
        let permuted_indices = indices.permute(&permute_axes)?;

        let inner_size = *permuted_indices.shape().last().unwrap();
        let outer_size = permuted_indices.size() as i64 / inner_size;

        let mut res_shape = self.shape().to_vec();
        res_shape[axis] = *permuted_indices.shape().last().unwrap();
        let res = Self::empty(res_shape)?;
        let permuted_self = self.permute(&permute_axes)?;

        if self.ndim() == 1 {
            for i in 0..indices.size() {
                let idx = indices.ptr()[i];
                if idx < 0 || idx >= self.shape()[0] as i64 {
                    return Err(anyhow::anyhow!("index out of bounds"));
                }
                res.ptr()[i] = self.ptr()[idx as usize];
            }
            return Ok(res);
        }

        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = (outer_size as usize).min(pool.max_count());
            let intervals = mt_intervals(outer_size as usize, num_threads);

            for (start, end) in intervals {
                let self_ptr = permuted_self.ptr();
                let indices_ptr = permuted_indices.ptr();
                let mut res_ptr = res.ptr();

                let gather_size = indices.shape()[axis];
                let self_last_strides = *self.strides().last().unwrap();
                let axis_self_last_stride = self.strides()[axis];
                let self_inner_size = *permuted_self.shape().last().unwrap();

                let res_shape = res.shape().clone();
                let res_strides = res.strides().clone();

                let permuted_self_shape = permuted_self.shape().clone();
                let permuted_self_strides = permuted_self.strides().clone();

                let indices_shape = indices.shape().clone();
                let indices_strides = indices.strides().clone();

                pool.execute(move || {
                    for outer_idx in start as i64..end as i64 {
                        let mut indices_amount = outer_idx * *indices_shape.last().unwrap();
                        let mut indices_offset = 0;
                        for j in (0..indices_shape.len()).rev() {
                            indices_offset +=
                                (indices_amount % indices_shape[j]) * indices_strides[j];
                            indices_amount /= indices_shape[j];
                        }

                        let mut res_indices_amount = outer_idx * gather_size;
                        let mut res_indices_offset = 0;
                        for j in (0..res_shape.len()).rev() {
                            res_indices_offset +=
                                (res_indices_amount % res_shape[j]) * res_strides[j];
                            res_indices_amount /= res_shape[j];
                        }

                        let mut inp_indices_amount = outer_idx * self_inner_size;
                        let mut inp_indices_offset = 0;
                        for j in (0..permuted_self_shape.len() - 1).rev() {
                            inp_indices_offset += (inp_indices_amount % permuted_self_shape[j])
                                * permuted_self_strides[j];
                            inp_indices_amount /= permuted_self_shape[j];
                        }
                        for gather_idx in 0..gather_size {
                            let index = indices_ptr[indices_offset + gather_idx];
                            let val = self_ptr[(inp_indices_offset
                                + gather_idx * self_last_strides)
                                + index * axis_self_last_stride];
                            res_ptr[res_indices_offset + gather_idx] = val;
                        }
                    }
                });
            }
            pool.join();
        });

        Ok(res)
    }
}

impl<T> Tensor<T, Cpu>
where
    T: CommonBounds,
{
    /// Gathers elements from the tensor along a specified axis using the provided indices.
    ///
    /// This method retrieves elements from the input tensor based on the positions specified in the `indices` tensor,
    /// along the specified `axis`. The shape of the output tensor corresponds to the shape of the `indices` tensor.
    /// This operation is useful for selecting specific elements along a given axis, such as when performing indexing
    /// or advanced slicing operations.
    ///
    /// # Arguments
    ///
    /// * `indices` - A tensor of type `i64` containing the indices that specify which elements to gather
    ///   from the input tensor along the given axis. The shape of the `indices` tensor determines the shape
    ///   of the output tensor.
    /// * `axis` - The axis along which to gather the elements. This must be a valid axis for the input tensor.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the gathered elements.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn gather(&self, indices: &Tensor<i64, Cpu>, axis: i64) -> anyhow::Result<Tensor<T>> {
        Ok(self.inner.gather(indices.inner.as_ref(), axis)?.into())
    }
}
