use crate::ops::cuda::cuda_utils::{
    compile_kernel, compute_kernel_launch_config, get_module_name_vec,
};
use crate::Cuda;
use crate::{ops::cuda::cuda_utils::get_include_1, tensor_base::_Tensor};
use cudarc::driver::{DeviceRepr, LaunchAsync};
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_common::error::param::ParamError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_common::slice;
use hpt_macros::select;
use hpt_traits::{CommonBounds, ShapeManipulate, Slice, TensorCreator, TensorInfo, TensorLike};
use hpt_types::dtype::CudaType;
use std::panic::Location;

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE: usize> ShapeManipulate
    for _Tensor<T, Cuda, DEVICE>
{
    type Meta = T;
    type Output = Self;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::squeeze(
            self,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::unsqueeze(
            self,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::reshape(
            self,
            shape,
            |a| a.contiguous(),
        )?)
    }
    fn transpose(&self, axis1: i64, axis2: i64) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::transpose(
            self, axis1, axis2,
        )?)
    }
    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::permute(
            self,
            axes,
            |layout, axes| layout.permute(axes),
        )?)
    }

    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::permute(
            self,
            axes,
            |layout, axes| layout.permute_inv(axes),
        )?)
    }
    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::expand(self, shape)?)
    }
    fn t(&self) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::t(self)?)
    }
    fn mt(&self) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::mt(self)?)
    }
    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::flip(self, axes)?)
    }
    fn fliplr(&self) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::fliplr(self)?)
    }
    fn flipud(&self) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::flipud(self)?)
    }
    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::tile(
            self,
            repeats,
            |a| a.contiguous(),
        )?)
    }
    fn trim_zeros(&self, trim: &str) -> Result<Self, TensorError>
    where
        Self::Meta: PartialEq,
    {
        ParamError::check_trim(trim)?;
        if self.ndim() > 1 {
            return Err(ShapeError::InvalidDimension {
                message: "trim_zeros only support 1D tensor".to_string(),
                location: Location::caller(),
            }
            .into());
        }
        let stride = self.strides()[0] as isize;
        let raw = self.as_raw();
        let mut ptr = raw.as_ptr();
        let mut left_len = 0;
        if trim.contains('f') {
            unsafe {
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        left_len += 1;
                    }
                }
            }
        }
        let mut right_len = raw.len() as i64;
        if trim.contains('b') {
            unsafe {
                ptr = raw.as_ptr().offset(((raw.len() - 1) as isize) * stride);
                let stride = -stride;
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        right_len -= 1;
                    }
                }
            }
        }
        slice!(self[left_len:right_len])
    }
    fn repeat(&self, repeats: usize, axes: i16) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::repeat(
            self,
            repeats,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::split(
            self,
            indices_or_sections,
            axis,
        )?)
    }
    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::dsplit(self, indices)?)
    }
    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::hsplit(self, indices)?)
    }
    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::vsplit(self, indices)?)
    }
    fn swap_axes(&self, axis1: i64, axis2: i64) -> Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::swap_axes(
            self, axis1, axis2,
        )?)
    }
    fn flatten<A>(&self, start_dim: A, end_dim: A) -> Result<Self, TensorError>
    where
        A: Into<Option<usize>>,
    {
        Ok(crate::ops::common::shape_manipulate::flatten(
            self,
            start_dim,
            end_dim,
            |a| a.contiguous(),
        )?)
    }
    fn concat(tensors: Vec<Self>, axis: usize, keepdims: bool) -> Result<Self, TensorError>
    where
        T: 'static,
    {
        let length = tensors.len();
        for i in tensors.iter() {
            for (idx, x) in tensors[0].shape().iter().enumerate() {
                if idx != axis
                    && i.shape().len() == tensors[0].shape().len()
                    && *x != i.shape()[idx]
                {
                    return Err(ShapeError::ConcatDimMismatch {
                        expected: *x as usize,
                        actual: i.shape()[idx] as usize,
                        location: Location::caller(),
                    }
                    .into());
                } else if i.shape().len() != tensors[0].shape().len() {
                    ShapeError::check_ndim_enough(
                        "concat dim mismatch".to_string(),
                        tensors[0].ndim(),
                        i.ndim(),
                    )?;
                }
            }
        }
        let mut new_shape: Vec<i64> = vec![0; tensors[0].ndim()];
        tensors.iter().for_each(|x| {
            new_shape[axis] += x.shape()[axis];
        });
        tensors[0].shape().iter().enumerate().for_each(|(i, x)| {
            if i != axis {
                new_shape[i] = *x;
            }
        });
        let new_tensor = _Tensor::<T, Cuda, DEVICE>::empty(&new_shape)?;
        let mut begin = 0;
        let mut res_slices = Vec::with_capacity(length);
        for i in tensors.iter() {
            let mut selections = vec![(0, 0, 0); new_shape.len()];
            selections[axis] = (begin, begin + i.shape()[axis], 1);
            begin += i.shape()[axis];
            let res_tensor = new_tensor.slice(&selections)?;
            res_slices.push(res_tensor);
        }
        let tensors = tensors
            .iter()
            .map(|x| (*x).clone())
            .collect::<Vec<_Tensor<T, Cuda, DEVICE>>>();

        let include = get_include_1::<T>();
        let module_name = get_module_name_vec("cc", &tensors);
        let map = compile_kernel(
            &module_name,
            &format!(
                "
                        {include}
                        extern \"C\" __global__ void assign({} *out, {} *inp, long long *shape, long long *strides, long long *inp_shape, long long *inp_strides, size_t ndim, size_t size)
                        {{
                            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                            size_t stride = blockDim.x * gridDim.x;
                            while (idx < size)
                            {{
                                long inp_amount = idx;
                                long inp_offset = 0;
                                long out_offset = 0;
                                long out_amount = idx;
                                for (int j = ndim - 1; j >= 0; j--)
                                {{
                                    inp_offset += (inp_amount % inp_shape[j]) * inp_strides[j];
                                    inp_amount /= inp_shape[j];
                                    out_offset += (out_amount % shape[j]) * strides[j];
                                    out_amount /= shape[j];
                                }}
                                out[out_offset] = inp[inp_offset];
                                idx += stride;
                            }}
                        }}",
                T::CUDA_TYPE,
                T::CUDA_TYPE,
            ),
            tensors[0].device(),
            &["assign"],
        )?;
        let kernel = tensors[0]
            .device()
            .get_func(&module_name, "assign")
            .unwrap();
        let reg_info = map.get("assign").expect("func_name not found");
        for (res, input) in res_slices.into_iter().zip(tensors.into_iter()) {
            let out_slice = res.cuda_slice();
            let inp_slice = input.cuda_slice();
            let inp_shape = new_tensor.cuda_shape()?;
            let inp_strides = new_tensor.cuda_strides()?;
            let shape = new_tensor.cuda_shape()?;
            let strides = new_tensor.cuda_strides()?;
            let cfg = compute_kernel_launch_config(res.device(), reg_info, input.size());
            unsafe {
                kernel.clone().launch(
                    cfg,
                    (
                        out_slice,
                        inp_slice,
                        &shape,
                        &strides,
                        &inp_shape,
                        &inp_strides,
                        input.ndim() as u64,
                        input.size() as u64,
                    ),
                )?;
            };
        }
        if keepdims {
            let mut res_shape = Vec::with_capacity(new_shape.len() + 1);
            for (idx, i) in new_shape.iter().enumerate() {
                if idx == axis {
                    res_shape.push(length as i64);
                    res_shape.push(*i / (length as i64));
                } else {
                    res_shape.push(*i);
                }
            }
            new_tensor.reshape(res_shape)
        } else {
            Ok(new_tensor)
        }
    }
    fn vstack(tensors: Vec<Self>) -> Result<Self, TensorError> {
        Self::concat(tensors, 0, false)
    }
    fn hstack(mut tensors: Vec<Self>) -> Result<Self, TensorError> {
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 2 {
                return if tensor.shape().len() == 1 {
                    Self::concat(tensors, 0, false)
                } else {
                    // scalar
                    let mut tensors_ref = Vec::with_capacity(tensors.len());
                    let mut tensors_holder = Vec::with_capacity(tensors.len());
                    for tensor in tensors {
                        tensors_holder.push(tensor.reshape(vec![1])?);
                    }
                    for tensor in tensors_holder {
                        tensors_ref.push(tensor);
                    }
                    Self::concat(tensors_ref, 0, false)
                };
            }
        }
        Self::concat(tensors, 1, false)
    }
    fn dstack(mut tensors: Vec<Self>) -> Result<Self, TensorError> {
        let mut new_tensors = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 3 {
                if tensor.shape().len() == 1 {
                    new_tensors.push(tensor.reshape(vec![1, tensor.shape()[0], 1])?);
                } else if tensor.shape().len() == 0 {
                    new_tensors.push(tensor.reshape(vec![1, 1, 1])?);
                } else {
                    new_tensors.push(tensor.reshape(vec![
                        tensor.shape()[0],
                        tensor.shape()[1],
                        1,
                    ])?);
                }
            } else {
                new_tensors.push(tensor.clone());
            }
        }
        let mut tensors_ref = Vec::with_capacity(new_tensors.len());
        for tensor in new_tensors {
            tensors_ref.push(tensor);
        }
        Self::concat(tensors_ref, 2, false)
    }
}
