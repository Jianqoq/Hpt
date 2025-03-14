use crate::common::Shape;
use crate::error::TensorError;
use crate::tensor_base::_Tensor;
use crate::ALIGN;
use crate::THREAD_POOL;
use hpt_allocator::traits::Allocator;
use hpt_allocator::traits::AllocatorOutputRetrive;
use hpt_allocator::Cpu;
use hpt_common::axis::axis::process_axes;
use hpt_common::axis::axis::Axis;
use hpt_common::error::param::ParamError;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::fft::FFTOps;
use hpt_traits::ops::shape_manipulate::ShapeManipulate;
use hpt_traits::ops::unary::Contiguous;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use num::complex::Complex32;
use num::complex::Complex64;
use num::Complex;
use num::FromPrimitive;
use std::ops::Div;
use std::panic::Location;
use std::sync::Arc;

pub(crate) fn fftn_template<T, A: Into<Axis>, S: Into<Shape>, const DEVICE: usize, Al>(
    x: &_Tensor<Complex<T>, Cpu, DEVICE, Al>,
    s: S,
    axes: A,
    norm: Option<&str>,
    backward: fn(Complex<T>, T) -> Complex<T>,
    forward: fn(Complex<T>, T) -> Complex<T>,
    ortho: fn(Complex<T>, T) -> Complex<T>,
    init_fft: fn(rustfft::FftPlanner<T>, usize) -> Arc<dyn rustfft::Fft<T>>,
) -> Result<_Tensor<Complex<T>, Cpu, DEVICE, Al>, TensorError>
where
    T: CommonBounds
        + FromPrimitive
        + num::Signed
        + Cast<T>
        + FloatOutUnary<Output = T>
        + FloatOutBinary<Output = T>,
    Complex<T>: Div<Output = Complex<T>> + CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    i64: Cast<T>,
{
    let s: Shape = s.into();
    let s = s.to_vec();
    // Process the axes and check for errors.
    let axes = process_axes(axes, x.ndim())?;
    // Create an empty tensor with the same properties as x.
    let mut res = x.clone();

    // Clone the original tensor to avoid modifying it directly.
    let mut self_clone = if x.is_contiguous() && x.parent().is_none() {
        x.clone()
    } else {
        x.contiguous()?
    };
    let post_op = if let Some(norm) = norm {
        match norm {
            "backward" => backward,
            "forward" => forward,
            "ortho" => ortho,
            _ => {
                return Err(ParamError::InvalidFFTNormParam {
                    value: norm.to_string(),
                    location: Location::caller(),
                }
                .into())
            }
        }
    } else {
        backward
    };
    let len = axes.len();
    let mut n = T::ZERO;
    // Iterate through each axis and apply FFT.
    for (idx, (axis, dim)) in axes.iter().zip(s.iter()).enumerate() {
        // Swap the current axis with the last dimension for easier processing.
        let mut new_axes = (0..self_clone.ndim() as i64).collect::<Vec<i64>>();
        new_axes.swap(*axis, self_clone.ndim() - 1);

        let mut new_shape = res.shape().to_vec();
        new_shape[*axis] = *dim;
        res = _Tensor::<Complex<T>, Cpu, DEVICE, Al>::empty(new_shape)?;
        let op = if idx == len - 1 {
            n = axes
                .iter()
                .map(|axis| res.shape()[*axis])
                .product::<i64>()
                .cast();
            post_op
        } else {
            fn _forward<T>(x: Complex<T>, _: T) -> Complex<T> {
                x
            }
            _forward
        };
        // Transpose the tensor and the result tensor for the current axis.
        let transposed_res = res.permute(&new_axes)?;
        let new_self = self_clone.permute(new_axes)?;
        // Determine the size of the inner and outer loops for parallel processing.
        let inner_loop_size = new_self.shape()[new_self.ndim() - 1];
        let res_inner_loop_size = transposed_res.shape()[transposed_res.ndim() - 1];
        let outer_loop_size = new_self.size() as i64 / inner_loop_size;
        let inp_last_stride = new_self.strides()[new_self.ndim() - 1];
        let out_last_stride = transposed_res.strides()[new_self.ndim() - 1];
        THREAD_POOL.with_borrow_mut(|pool| {
            unsafe {
                let num_threads;
                if outer_loop_size < pool.max_count() as i64 {
                    num_threads = outer_loop_size as usize;
                } else {
                    num_threads = pool.max_count();
                }
                let ndim = new_self.ndim();
                let mut shape = new_self.shape().to_vec();
                shape.iter_mut().for_each(|x| {
                    *x -= 1;
                });

                let intervals = mt_intervals(outer_loop_size as usize, num_threads);

                let mut prgs = Vec::with_capacity(num_threads);
                let mut ptrs = Vec::with_capacity(num_threads);
                let mut res_ptrs = Vec::with_capacity(num_threads);

                for i in 0..num_threads {
                    let mut local_ptr = new_self.data.clone();
                    let mut local_res_ptr = transposed_res.data.clone();
                    let mut amount = intervals[i].0 as i64 * inner_loop_size;
                    let mut res_amount = intervals[i].0 as i64 * res_inner_loop_size;
                    let mut current_prg = vec![0; ndim];
                    for j in (0..ndim).rev() {
                        current_prg[j] = amount % new_self.shape()[j];
                        local_ptr += current_prg[j] * new_self.strides()[j];
                        local_res_ptr +=
                            (res_amount % transposed_res.shape()[j]) * transposed_res.strides()[j];
                        amount /= new_self.shape()[j];
                        res_amount /= transposed_res.shape()[j];
                    }
                    prgs.push(current_prg);
                    ptrs.push(local_ptr);
                    res_ptrs.push(local_res_ptr);
                }
                for idx in (0..num_threads).rev() {
                    let local_shape = shape.clone();
                    let res_shape = transposed_res.shape().clone();
                    let mut local_prg = prgs.pop().unwrap();
                    let mut ptr = ptrs.pop().unwrap();
                    let mut res_ptr = res_ptrs.pop().unwrap();
                    let current_size = intervals[idx].1 - intervals[idx].0;

                    let planner = rustfft::FftPlanner::<T>::new();

                    let inner_fft = init_fft(planner, res_inner_loop_size as usize);
                    let strides = new_self.strides().clone();
                    let trasposed_strides = transposed_res.strides().clone();
                    pool.execute(move || {
                        // Allocate buffer and perform FFT in parallel. (Could be removed? directly use result Tensor as buffer)
                        let raw_buffer = std::alloc::alloc(
                            std::alloc::Layout::from_size_align(
                                (res_inner_loop_size as usize) * std::mem::size_of::<Complex<T>>(),
                                ALIGN,
                            )
                            .unwrap(),
                        );
                        let buffer: &mut [Complex<T>] = std::slice::from_raw_parts_mut(
                            raw_buffer as *mut Complex<T>,
                            res_inner_loop_size as usize,
                        );
                        for _ in 0..current_size {
                            if res_inner_loop_size > inner_loop_size {
                                for i in 0..inner_loop_size {
                                    buffer[i as usize] = ptr[i * inp_last_stride];
                                }
                                for i in inner_loop_size..res_inner_loop_size {
                                    buffer[i as usize] = Complex::<T>::ZERO;
                                }
                            } else {
                                for i in 0..res_inner_loop_size {
                                    buffer[i as usize] = ptr[i * inp_last_stride];
                                }
                            }
                            inner_fft.process(buffer);
                            for i in 0..res_inner_loop_size {
                                res_ptr[i * out_last_stride] = op(buffer[i as usize], n);
                            }
                            for j in (0..ndim - 1).rev() {
                                if local_prg[j] < local_shape[j] {
                                    local_prg[j] += 1;
                                    ptr += strides[j];
                                    res_ptr += trasposed_strides[j];
                                    break;
                                } else {
                                    local_prg[j] = 0;
                                    ptr -= strides[j] * local_shape[j];
                                    res_ptr -= trasposed_strides[j] * (res_shape[j] - 1);
                                }
                            }
                        }
                        std::alloc::dealloc(
                            raw_buffer as *mut u8,
                            std::alloc::Layout::from_size_align(
                                (res_inner_loop_size as usize) * std::mem::size_of::<T>(),
                                ALIGN,
                            )
                            .unwrap(),
                        );
                    });
                }
                pool.join();
            }
        });
        self_clone = res.clone();
    }
    Ok(res)
}

macro_rules! impl_fftops {
    ($type:ident, $meta_type:ident) => {
        impl FFTOps for _Tensor<$type> {
            fn fft(&self, n: usize, axis: i64, norm: Option<&str>) -> Result<Self, TensorError> {
                self.fftn(&[n], axis, norm)
            }
            fn ifft(&self, n: usize, axis: i64, norm: Option<&str>) -> Result<Self, TensorError> {
                self.ifftn(&[n], axis, norm)
            }
            fn fft2<S: Into<Shape>>(
                &self,
                s: S,
                axis1: i64,
                axis2: i64,
                norm: Option<&str>,
            ) -> Result<Self, TensorError> {
                self.fftn(s, [axis1, axis2], norm)
            }
            fn ifft2<S: Into<Shape>>(
                &self,
                s: S,
                axis1: i64,
                axis2: i64,
                norm: Option<&str>,
            ) -> Result<Self, TensorError> {
                self.ifftn(s, [axis1, axis2], norm)
            }
            fn fftn<A: Into<Axis>, S: Into<Shape>>(
                &self,
                s: S,
                axes: A,
                norm: Option<&str>,
            ) -> Result<Self, TensorError> {
                fn backward(x: $type, _: $meta_type) -> $type {
                    x
                }
                fn forward(x: $type, n: $meta_type) -> $type {
                    x / $type::new(n, <$meta_type>::ZERO)
                }
                fn ortho(x: $type, n: $meta_type) -> $type {
                    x / $type::from(n.sqrt())
                }
                fftn_template(
                    self,
                    s,
                    axes,
                    norm,
                    backward,
                    forward,
                    ortho,
                    |mut planner, size| planner.plan_fft_forward(size),
                )
            }
            fn ifftn<A: Into<Axis>, S: Into<Shape>>(
                &self,
                s: S,
                axes: A,
                norm: Option<&str>,
            ) -> Result<Self, TensorError> {
                fn backward(x: $type, n: $meta_type) -> $type {
                    x / $type::from(n)
                }
                fn forward(x: $type, _: $meta_type) -> $type {
                    x
                }
                fn ortho(x: $type, n: $meta_type) -> $type {
                    x / $type::from(n.sqrt())
                }
                fftn_template(
                    self,
                    s,
                    axes,
                    norm,
                    backward,
                    forward,
                    ortho,
                    |mut planner, size| planner.plan_fft_inverse(size),
                )
            }
        }
    };
}

impl_fftops!(Complex32, f32);
impl_fftops!(Complex64, f64);
