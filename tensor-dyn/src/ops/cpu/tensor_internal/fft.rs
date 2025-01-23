// use crate::tensor_base::_Tensor;
// use crate::ALIGN;
// use crate::THREAD_POOL;
// use num::complex::Complex32;
// use num::complex::Complex64;
// use std::sync::Arc;
// use std::sync::Barrier;
// use tensor_common::axis::process_axes;
// use tensor_common::axis::Axis;
// use tensor_common::shape::shape_utils::mt_intervals;
// use tensor_traits::ops::fft::FFTOps;
// use tensor_traits::shape_manipulate::ShapeManipulate;
// use tensor_traits::tensor::TensorCreator;
// use tensor_traits::tensor::TensorInfo;

// macro_rules! impl_fftops {
//     ($type:ident, $meta_type:ident) => {
//         impl FFTOps for _Tensor<$type> {
//             fn fft(&self, axis: i64) -> anyhow::Result<Self> {
//                 self.fftn(axis)
//             }
//             fn ifft(&self, axis: i64) -> anyhow::Result<Self> {
//                 self.ifftn(axis)
//             }
//             fn fft2(&self, axis1: i64, axis2: i64) -> anyhow::Result<Self> {
//                 self.fftn([axis1, axis2])
//             }
//             fn ifft2(&self, axis1: i64, axis2: i64) -> anyhow::Result<Self> {
//                 self.ifftn([axis1, axis2])
//             }
//             fn fftn<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Self> {
//                 // Process the axes and check for errors.
//                 let axes = process_axes(axes, self.ndim())?;
//                 // Create an empty tensor with the same properties as self.
//                 let res = self.empty_like()?;

//                 // Clone the original tensor to avoid modifying it directly.
//                 let mut self_clone = self.clone();

//                 // Iterate through each axis and apply FFT.
//                 for (idx, axis) in axes.into_iter().enumerate() {
//                     // Swap the current axis with the last dimension for easier processing.
//                     let mut new_axes = (0..self_clone.ndim() as i64).collect::<Vec<i64>>();
//                     new_axes.swap(axis, self_clone.ndim() - 1);

//                     // Transpose the tensor and the result tensor for the current axis.
//                     let transposed_res = res.permute(&new_axes)?;
//                     let new_self = self_clone.permute(new_axes)?;

//                     // Determine the size of the inner and outer loops for parallel processing.
//                     let inner_loop_size = new_self.shape()[new_self.ndim() - 1];
//                     let outer_loop_size = new_self.size() as i64 / inner_loop_size;
//                     let last_stride = new_self.strides()[new_self.ndim() - 1];

//                     THREAD_POOL.with_borrow_mut(|pool| {
//                         unsafe {
//                             let num_threads;
//                             if outer_loop_size < pool.max_count() as i64 {
//                                 num_threads = outer_loop_size as usize;
//                             } else {
//                                 num_threads = pool.max_count();
//                             }
//                             let ndim = new_self.ndim();
//                             let mut shape = new_self.shape().to_vec();
//                             shape.iter_mut().for_each(|x| {
//                                 *x -= 1;
//                             });

//                             let intervals = mt_intervals(outer_loop_size as usize, num_threads);

//                             let mut prgs = Vec::with_capacity(num_threads);
//                             let mut ptrs = Vec::with_capacity(num_threads);
//                             let mut res_ptrs = Vec::with_capacity(num_threads);
//                             let mut amount = 0;

//                             for i in 0..num_threads {
//                                 let mut local_ptr = new_self.data.clone();
//                                 let mut local_res_ptr = transposed_res.data.clone();

//                                 let mut local_amount = amount;

//                                 let mut current_prg = vec![0; ndim];
//                                 for j in (0..ndim).rev() {
//                                     current_prg[j] = local_amount % new_self.shape()[j];
//                                     local_ptr.offset(current_prg[j] * new_self.strides()[j]);
//                                     local_res_ptr
//                                         .offset(current_prg[j] * transposed_res.strides()[j]);
//                                     local_amount /= new_self.shape()[j];
//                                 }
//                                 amount +=
//                                     ((intervals[i].1 - intervals[i].0) as i64) * inner_loop_size;
//                                 prgs.push(current_prg);
//                                 ptrs.push(local_ptr);
//                                 res_ptrs.push(local_res_ptr);
//                             }
//                             let barrier = Arc::new(Barrier::new(num_threads + 1));
//                             for idx in (0..num_threads).rev() {
//                                 // Allocate buffer and perform FFT in parallel. (Could be removed?)
//                                 let raw_buffer = std::alloc::alloc(
//                                     std::alloc::Layout::from_size_align(
//                                         (inner_loop_size as usize) * std::mem::size_of::<$type>(),
//                                         ALIGN,
//                                     )
//                                     .unwrap(),
//                                 );
//                                 let buffer: &mut [$type] = std::slice::from_raw_parts_mut(
//                                     raw_buffer as *mut $type,
//                                     inner_loop_size as usize,
//                                 );

//                                 let local_shape = shape.clone();
//                                 let mut local_prg = prgs.pop().unwrap();
//                                 let mut ptr = ptrs.pop().unwrap();
//                                 let mut res_ptr = res_ptrs.pop().unwrap();
//                                 let current_size = intervals[idx].1 - intervals[idx].0;

//                                 let mut planner = rustfft::FftPlanner::<$meta_type>::new();

//                                 let inner_fft = planner.plan_fft_forward(inner_loop_size as usize);
//                                 let strides = new_self.strides().clone();
//                                 let trasposed_strides = transposed_res.strides().clone();
//                                 let barrier_clone = barrier.clone();
//                                 pool.execute(move || {
//                                     for _ in 0..current_size {
//                                         for i in 0..inner_loop_size {
//                                             buffer[i as usize] = ptr[i * last_stride];
//                                         }
//                                         inner_fft.process(buffer);
//                                         for i in 0..inner_loop_size {
//                                             res_ptr.modify(i * last_stride, buffer[i as usize]);
//                                         }
//                                         for j in (0..ndim - 1).rev() {
//                                             if local_prg[j] < local_shape[j] {
//                                                 local_prg[j] += 1;
//                                                 ptr.offset(strides[j]);
//                                                 res_ptr.offset(trasposed_strides[j]);
//                                                 break;
//                                             } else {
//                                                 local_prg[j] = 0;
//                                                 ptr.offset(-strides[j] * local_shape[j]);
//                                                 res_ptr
//                                                     .offset(-trasposed_strides[j] * local_shape[j]);
//                                             }
//                                         }
//                                     }
//                                     barrier_clone.wait();
//                                 });
//                                 std::alloc::dealloc(
//                                     raw_buffer as *mut u8,
//                                     std::alloc::Layout::from_size_align(
//                                         (inner_loop_size as usize) * std::mem::size_of::<$type>(),
//                                         ALIGN,
//                                     )
//                                     .unwrap(),
//                                 );
//                             }
//                             barrier.wait();
//                             if idx == 0 {
//                                 self_clone = res.clone();
//                             }
//                         }
//                     });
//                 }
//                 Ok(res)
//             }
//             fn ifftn<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Self> {
//                 let axes = process_axes(axes, self.ndim())?;
//                 let res = self.empty_like()?;

//                 let mut self_clone = self.clone();

//                 for (idx, axis) in axes.into_iter().enumerate() {
//                     let mut new_axes = (0..self_clone.ndim() as i64).collect::<Vec<i64>>();
//                     new_axes.swap(axis, self_clone.ndim() - 1);
//                     let transposed_res = res.permute(&new_axes)?;
//                     let new_self = self_clone.permute(new_axes)?;
//                     let inner_loop_size = new_self.shape()[new_self.ndim() - 1];
//                     let outer_loop_size = (new_self.size() as i64) / inner_loop_size;
//                     let last_stride = new_self.strides()[new_self.ndim() - 1];

//                     THREAD_POOL.with_borrow_mut(|pool| unsafe {
//                         let num_threads;
//                         if outer_loop_size < (pool.max_count() as i64) {
//                             num_threads = outer_loop_size as usize;
//                         } else {
//                             num_threads = pool.max_count();
//                         }
//                         let ndim = new_self.ndim();
//                         let mut shape = new_self.shape().to_vec();
//                         shape.iter_mut().for_each(|x| {
//                             *x -= 1;
//                         });

//                         let intervals = mt_intervals(outer_loop_size as usize, num_threads);

//                         let mut prgs = Vec::with_capacity(num_threads);
//                         let mut ptrs = Vec::with_capacity(num_threads);
//                         let mut res_ptrs = Vec::with_capacity(num_threads);
//                         let mut amount = 0;

//                         for i in 0..num_threads {
//                             let mut local_ptr = new_self.data.clone();
//                             let mut local_res_ptr = transposed_res.data.clone();

//                             let mut local_amount = amount;

//                             let mut current_prg = vec![0; ndim];
//                             for j in (0..ndim).rev() {
//                                 current_prg[j] = local_amount % new_self.shape()[j];
//                                 local_ptr.offset(current_prg[j] * new_self.strides()[j]);
//                                 local_res_ptr.offset(current_prg[j] * transposed_res.strides()[j]);
//                                 local_amount /= new_self.shape()[j];
//                             }
//                             amount += ((intervals[i].1 - intervals[i].0) as i64) * inner_loop_size;
//                             prgs.push(current_prg);
//                             ptrs.push(local_ptr);
//                             res_ptrs.push(local_res_ptr);
//                         }
//                         let barrier = Arc::new(Barrier::new(num_threads + 1));
//                         for idx in (0..num_threads).rev() {
//                             let raw_buffer = std::alloc::alloc(
//                                 std::alloc::Layout::from_size_align(
//                                     (inner_loop_size as usize) * std::mem::size_of::<$type>(),
//                                     ALIGN,
//                                 )
//                                 .unwrap(),
//                             );
//                             let buffer: &mut [$type] = std::slice::from_raw_parts_mut(
//                                 raw_buffer as *mut $type,
//                                 inner_loop_size as usize,
//                             );
//                             let local_shape = shape.clone();
//                             let mut local_prg = prgs.pop().unwrap();
//                             let mut ptr = ptrs.pop().unwrap();
//                             let mut res_ptr = res_ptrs.pop().unwrap();
//                             let current_size = intervals[idx].1 - intervals[idx].0;

//                             let mut planner = rustfft::FftPlanner::<$meta_type>::new();

//                             let inner_fft = planner.plan_fft_inverse(inner_loop_size as usize);
//                             let strides = new_self.strides().clone();
//                             let trasposed_strides = transposed_res.strides().clone();
//                             let barrier_clone = barrier.clone();
//                             pool.execute(move || {
//                                 for _ in 0..current_size {
//                                     for i in 0..inner_loop_size {
//                                         buffer[i as usize] = ptr[i * last_stride];
//                                     }
//                                     inner_fft.process(buffer);
//                                     for i in 0..inner_loop_size {
//                                         res_ptr.modify(i * last_stride, buffer[i as usize]);
//                                     }
//                                     for j in (0..ndim - 1).rev() {
//                                         if local_prg[j] < local_shape[j] {
//                                             local_prg[j] += 1;
//                                             ptr.offset(strides[j]);
//                                             res_ptr.offset(trasposed_strides[j]);
//                                             break;
//                                         } else {
//                                             local_prg[j] = 0;
//                                             ptr.offset(-strides[j] * local_shape[j]);
//                                             res_ptr.offset(-trasposed_strides[j] * local_shape[j]);
//                                         }
//                                     }
//                                 }
//                                 barrier_clone.wait();
//                             });
//                             std::alloc::dealloc(
//                                 raw_buffer as *mut u8,
//                                 std::alloc::Layout::from_size_align(
//                                     (inner_loop_size as usize) * std::mem::size_of::<$type>(),
//                                     ALIGN,
//                                 )
//                                 .unwrap(),
//                             );
//                         }
//                         barrier.wait();
//                         if idx == 0 {
//                             self_clone = res.clone();
//                         }
//                     });
//                 }
//                 Ok(res)
//             }
//         }
//     };
// }

// impl_fftops!(Complex32, f32);
// impl_fftops!(Complex64, f64);
