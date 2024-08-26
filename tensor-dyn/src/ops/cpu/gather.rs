use std::sync::{ Arc, Barrier };

use tensor_common::{ shape_utils::mt_intervals, slice::Slice };
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };

use crate::{ backend::Cpu, slice::SliceOps, tensor_base::_Tensor, THREAD_POOL };

impl<T> _Tensor<T, Cpu> where T: CommonBounds {
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn gather(&self, indices: &_Tensor<i64, Cpu>, axis: i64) -> anyhow::Result<Self> {
        // assert_eq!(indices.ndim(), 1);
        // let axis = (if axis < 0 { (self.ndim() as i64) + axis } else { axis }) as usize;
        // let res_shape = self
        //     .shape()
        //     .iter()
        //     .enumerate()
        //     .map(|(i, &x)| if i == axis { indices.size() as i64 } else { x })
        //     .collect::<Vec<_>>();
        // let ret = _Tensor::<T, Cpu>::empty(res_shape)?;

        // THREAD_POOL.with_borrow_mut(|pool| {
        //     let num_threads = if indices.size() < pool.max_count() {
        //         indices.size()
        //     } else {
        //         pool.max_count()
        //     };
        //     let intervals = mt_intervals(indices.size(), num_threads);
        //     let mut sliced_res = Vec::with_capacity(num_threads);
        //     let mut sliced_indices = Vec::with_capacity(num_threads);
        //     for (start, end) in intervals.iter() {
        //         let mut slices = vec![Slice::Full; ret.ndim()];
        //         slices[axis] = Slice::Range((*start as i64, *end as i64));
        //         let sliced = ret.slice(&slices).expect("slice failed");
        //         sliced_res.push(sliced);
        //         let sliced_indice = indices
        //             .slice(&[Slice::Range((*start as i64, *end as i64))])
        //             .expect("slice failed");
        //         sliced_indices.push(sliced_indice);
        //     }
        //     let barrier = Arc::new(Barrier::new(num_threads + 1));
        //     for (res, indices) in sliced_res.into_iter().zip(sliced_indices.into_iter()) {
        //         let inp = self.clone();
        //         let barrier_clone = barrier.clone();
        //         pool.execute(move || {
        //             let mut slices = vec![Slice::Full; inp.ndim()];
        //             let mut res_slices = vec![Slice::Full; res.ndim()];
        //             let raw = indices.as_raw();
        //             for (i, idx) in raw.into_iter().enumerate() {
        //                 slices[axis] = Slice::Range((*idx, *idx + 1));
        //                 let slice = inp.slice(&slices).expect("slice failed");
        //                 res_slices[axis] = Slice::Range((i as i64, (i as i64) + 1));
        //                 let res_slice = res.slice(&res_slices).expect("slice failed");
        //                 res_slice
        //                     .iter_mut()
        //                     .zip(slice.iter())
        //                     .for_each(|(a, b)| {
        //                         *a = b;
        //                     }, |(a, b)| {
        //                         *a = b;
        //                     });
        //             }
        //             barrier_clone.wait();
        //         });
        //     }
        //     barrier.wait();
        // });
        // Ok(ret)
        todo!()
    }
}
