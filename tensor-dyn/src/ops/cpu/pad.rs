use std::sync::Arc;

use tensor_common::shape::shape_utils::mt_intervals;
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };

use crate::{ tensor_base::_Tensor, Tensor, THREAD_POOL };

impl<T: CommonBounds> _Tensor<T> {
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn pad(&self, pads: &[(i64, i64)], val: T) -> anyhow::Result<_Tensor<T>> {
        let res_shape = self
            .shape()
            .iter()
            .zip(pads.iter())
            .map(|(x, (a, b))| x + a + b)
            .collect::<Vec<_>>();

        let res = _Tensor::<T>::full(val, &res_shape)?;

        let outer_loop = self.shape()[..self.ndim() - 1].iter().product::<i64>();
        let inner_loop = self.shape()[self.ndim() - 1];
        let mut pads = pads.to_vec();
        if pads.len() < self.ndim() {
            pads.resize(self.ndim(), (0, 0));
        }
        let pads = Arc::new(pads);
        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if outer_loop < (pool.max_count() as i64) {
                outer_loop as usize
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(outer_loop as usize, num_threads);
            let mut ptrs = vec![];
            let mut prgs = vec![];
            for (start, _) in intervals.iter() {
                let mut ptr = self.ptr();
                let mut inp_index = 0;
                let mut inp_prg = vec![0; self.ndim()];
                let mut inp_amount = start * (self.shape()[self.ndim() - 1] as usize);
                for j in (0..self.ndim()).rev() {
                    inp_prg[j] = (inp_amount as i64) % self.shape()[j];
                    inp_amount /= self.shape()[j] as usize;
                    inp_index += inp_prg[j] * self.strides()[j];
                }
                ptr.offset(inp_index);
                ptrs.push(ptr);
                prgs.push(inp_prg);
            }
            for ((mut ptr, mut inp_prg), (start, end)) in ptrs
                .into_iter()
                .zip(prgs.into_iter())
                .zip(intervals.into_iter()) {
                let ndim = self.ndim();
                let ts = self.strides().clone();
                let tsp = self.shape().clone();
                let rs = res.strides().clone();
                let mut res_ptr = res.ptr();
                let inp_last_stride = self.strides()[ndim - 1];
                let pads = pads.clone();
                pool.execute(move || {
                    assert_eq!(tsp.len(), ndim);
                    assert_eq!(ts.len(), ndim);
                    assert_eq!(rs.len(), ndim);
                    for (dim_idx, (&prg_idx, &stride)) in inp_prg
                        .iter()
                        .zip(rs.iter())
                        .enumerate() {
                        let padded_idx = prg_idx + pads[dim_idx].0;
                        let offset = padded_idx * stride;
                        res_ptr.offset(offset);
                    }
                    for _ in start..end {
                        for i in 0..inner_loop {
                            res_ptr[i] = ptr[i * inp_last_stride];
                        }
                        for j in (0..ndim - 1).rev() {
                            let j = j;
                            if inp_prg[j] < tsp[j] - 1 {
                                inp_prg[j] += 1;
                                ptr.offset(ts[j]);
                                res_ptr.offset(rs[j]);
                                break;
                            } else {
                                inp_prg[j] = 0;
                                ptr.offset(-ts[j] * (tsp[j] - 1));
                                res_ptr.offset(-rs[j] * (tsp[j] - 1));
                            }
                        }
                    }
                });
            }
            pool.join();
        });

        Ok(res)
    }
}

impl<T: CommonBounds> Tensor<T> {
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn pad(&self, pads: &[(i64, i64)], val: T) -> anyhow::Result<Tensor<T>> {
        Ok(self.inner.pad(pads, val)?.into())
    }
}