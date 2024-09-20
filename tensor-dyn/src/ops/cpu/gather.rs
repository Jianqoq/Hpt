use std::sync::{Arc, Barrier};

use tensor_common::{shape_utils::mt_intervals, slice::Slice};
use tensor_iterator::{iterator_traits::StridedIterator, TensorIterator};
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo, TensorLike};

use crate::{backend::Cpu, tensor_base::_Tensor, THREAD_POOL};

impl<T> _Tensor<T, Cpu>
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
    pub fn gather(&self, indices: &_Tensor<i64, Cpu>, axis: i64) -> anyhow::Result<Self> {
        assert_eq!(indices.ndim(), 1);
        let axis = (if axis < 0 {
            (self.ndim() as i64) + axis
        } else {
            axis
        }) as usize;
        let res_shape = self
            .shape()
            .iter()
            .enumerate()
            .map(|(i, &x)| if i == axis { indices.size() as i64 } else { x })
            .collect::<Vec<_>>();
        let ret = _Tensor::<T, Cpu>::empty(res_shape)?;

        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if indices.size() < pool.max_count() {
                indices.size()
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(indices.size(), num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            let mut sliced_indices = Vec::with_capacity(num_threads);
            for (start, end) in intervals.iter() {
                let mut slices = vec![Slice::Full; ret.ndim()];
                slices[axis] = Slice::Range((*start as i64, *end as i64));
                let sliced = ret.slice(&slices).expect("slice failed");
                sliced_res.push(sliced);
                let sliced_indice = indices
                    .slice(&[Slice::Range((*start as i64, *end as i64))])
                    .expect("slice failed");
                sliced_indices.push(sliced_indice);
            }
            let barrier = Arc::new(Barrier::new(num_threads + 1));
            for (res, indices) in sliced_res.into_iter().zip(sliced_indices.into_iter()) {
                let inp = self.clone();
                let barrier_clone = barrier.clone();
                pool.execute(move || {
                    let mut slices = vec![Slice::Full; inp.ndim()];
                    let mut res_slices = vec![Slice::Full; res.ndim()];
                    let raw = indices.as_raw();
                    for (i, idx) in raw.into_iter().enumerate() {
                        slices[axis] = Slice::Range((*idx, *idx + 1));
                        let slice = inp.slice(&slices).expect("slice failed");
                        res_slices[axis] = Slice::Range((i as i64, (i as i64) + 1));
                        let res_slice = res.slice(&res_slices).expect("slice failed");
                        res_slice.iter_mut().zip(slice.iter()).for_each(|(a, b)| {
                            *a = b;
                        });
                    }
                    barrier_clone.wait();
                });
            }
            barrier.wait();
        });
        Ok(ret)
    }
}
