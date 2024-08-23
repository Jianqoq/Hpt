use std::sync::{ Arc, Barrier };

use tensor_common::shape_utils::mt_intervals;
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };

use crate::{ tensor_base::_Tensor, THREAD_POOL };

impl<T: CommonBounds> _Tensor<T> {
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
            let barrier = Arc::new(Barrier::new(num_threads + 1));
            for ((mut ptr, mut inp_prg), (start, end)) in ptrs
                .into_iter()
                .zip(prgs.into_iter())
                .zip(intervals.into_iter()) {
                let barrier_clone = barrier.clone();
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
                    for (dim_idx, (&prg_idx, &stride)) in inp_prg.iter().zip(rs.iter()).enumerate() {
                        let padded_idx = prg_idx + pads[dim_idx].0;
                        let offset = padded_idx * stride;
                        res_ptr.offset(offset);
                    }
                    for _ in start..end {
                        for i in 0..inner_loop {
                            res_ptr[i] = ptr[i * inp_last_stride];
                        }
                        for j in (0..ndim - 1).rev() {
                            let j = j as usize;
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
                    barrier_clone.wait();
                });
            }
            barrier.wait();
        });

        Ok(res)
    }
}
