use std::cell::RefCell;
use std::rc::Rc;

use crate::ops::diff::utils::handle_grad;
use crate::tensor::DiffTensor;
use crate::tensor::Tensor;
use crate::Cpu;
use tensor_common::error::base::TensorError;
use tensor_common::error::shape::ShapeError;
use tensor_common::slice::Slice;
use tensor_common::{axis::axis::Axis, shape::shape::Shape};
use tensor_iterator::iterator_traits::ParStridedIteratorZip;
use tensor_iterator::TensorIterator;
use tensor_traits::{CommonBounds, NormalReduce, ShapeManipulate, TensorCreator, TensorInfo};

impl<T: CommonBounds, const DEVICE: usize> ShapeManipulate for DiffTensor<T, Cpu, DEVICE> {
    type Meta = T;
    type Output = DiffTensor<T, Cpu, DEVICE>;

    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let res = self.inner.squeeze(axes.clone())?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.unsqueeze(axes.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let res = self.inner.unsqueeze(axes.clone())?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.squeeze(axes.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, TensorError> {
        let shape: Shape = shape.into();
        let original_shape = self.inner.shape().clone();
        let res = self.inner.reshape(shape.clone())?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.reshape(original_shape.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.transpose(axis1, axis2)?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.transpose(axis1, axis2)?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let res = self.inner.permute(axes.clone())?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.permute_inv(axes.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn permute_inv<A: Into<Axis>>(
        &self,
        axes: A,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let res = self.inner.permute_inv(axes.clone())?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.permute(axes.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, TensorError> {
        let shape: Shape = shape.into();
        let res = self.inner.expand(shape.clone())?;
        let mut lhs = self.clone();
        let mut sum_axes = Vec::new();
        for (i, dim) in self.inner.shape().iter().enumerate() {
            if *dim == 1 {
                sum_axes.push(i);
            }
        }
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.sum(sum_axes.clone(), true)?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn t(&self) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.t()?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.t()?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn mt(&self) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.mt()?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.mt()?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let res = self.inner.flip(axes.clone())?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.flip(axes.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn fliplr(&self) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.fliplr()?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.fliplr()?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn flipud(&self) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.flipud()?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.flipud()?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn tile<S: Into<Axis>>(&self, repeats: S) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.tile(repeats)?;
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| {
                unimplemented!("tile diff is not implemented yet");
            })),
        })
    }

    fn trim_zeros(&self, trim: &str) -> std::result::Result<Self::Output, TensorError>
    where
        Self::Meta: PartialEq,
    {
        let res = self.inner.trim_zeros(trim)?;
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| {
                unimplemented!("trim_zeros diff is not implemented yet");
            })),
        })
    }

    fn repeat(&self, repeats: usize, axes: i16) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.repeat(repeats, axes)?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.sum((axes + 1) as i64, false)?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> std::result::Result<Vec<Self>, TensorError> {
        use rayon::iter::ParallelIterator;
        let mut slices = Vec::with_capacity(self.inner.ndim());
        let mut tmp = Vec::with_capacity(self.inner.ndim());
        for _ in 0..self.inner.ndim() {
            tmp.push(Slice::Full);
        }
        let mut prev = 0;
        for &i in indices_or_sections.iter() {
            tmp[axis as usize] = Slice::Range((prev, i));
            prev = i;
            slices.push(tmp.clone());
        }
        let last = *indices_or_sections.last().unwrap();
        tmp[axis as usize] = Slice::Range((last, self.inner.shape()[axis as usize]));
        slices.push(tmp);
        let res = self.inner.split(indices_or_sections, axis)?;

        *self.grad.borrow_mut() = Some(self.inner.empty_like()?);
        let mut rets = Vec::with_capacity(res.len());
        for (idx, (splited, slice)) in res.into_iter().zip(slices).enumerate() {
            let lhs = self.clone();
            let diff =
                DiffTensor {
                    inner: splited,
                    grad: Rc::new(RefCell::new(None)),
                    out_degree: Rc::new(RefCell::new(0)),
                    backward: if idx == indices_or_sections.len() - 1 {
                        Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                            let taked = lhs.grad.borrow_mut().take();
                            if let Some(taked) = taked {
                                let sliced = taked.inner.slice(&slice)?;
                                sliced.par_iter_mut().zip(grad.inner.par_iter()).for_each(
                                    |(a, b)| {
                                        *a = b;
                                    },
                                );
                                let res = lhs.backward.borrow_mut()(taked.clone())?;
                                if res {
                                    lhs.grad.borrow_mut().replace(taked);
                                }
                            } else {
                                unreachable!()
                            }
                            Ok(false)
                        }))
                    } else {
                        Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                            let taked = lhs.grad.borrow_mut().take();
                            if let Some(taked) = taked {
                                let sliced = taked.inner.slice(&slice)?;
                                sliced.par_iter_mut().zip(grad.inner.par_iter()).for_each(
                                    |(a, b)| {
                                        *a = b;
                                    },
                                );
                                lhs.grad.borrow_mut().replace(taked);
                            } else {
                                unreachable!()
                            }
                            Ok(false)
                        }))
                    },
                };
            rets.push(diff);
        }
        Ok(rets)
    }

    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        ShapeError::check_ndim_enough(3, self.inner.ndim())?;
        DiffTensor::split(self, indices, 2)
    }

    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        ShapeError::check_ndim_enough(2, self.inner.ndim())?;
        DiffTensor::split(self, indices, 1)
    }

    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        ShapeError::check_ndim_enough(1, self.inner.ndim())?;
        DiffTensor::split(self, indices, 0)
    }

    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.swap_axes(axis1, axis2)?;
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.swap_axes(axis1, axis2)?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn flatten<A>(&self, start: A, end: A) -> std::result::Result<Self::Output, TensorError>
    where
        A: Into<Option<usize>>,
    {
        let res = self.inner.flatten(start, end)?;
        let mut lhs = self.clone();
        let original_shape = self.inner.shape().clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grad = grad.reshape(original_shape.clone())?;
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn concat(
        tensors: Vec<&Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let mut inners = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter() {
            inners.push(&tensor.inner);
        }
        let mut begin = 0;
        let mut split_sizes = Vec::with_capacity(tensors.len());
        for i in tensors.iter() {
            begin += i.inner.shape()[axis];
            split_sizes.push(begin);
        }
        let res = Tensor::concat(inners, axis, keepdims)?;
        let tensors = tensors.into_iter().map(|x| x.clone()).collect::<Vec<_>>();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let grads = grad.split(&split_sizes, axis as i64)?;
                for (idx, grad) in grads.into_iter().enumerate() {
                    let mut lhs = tensors[idx].clone();
                    handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                }
                Ok(false)
            })),
        })
    }

    fn vstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, TensorError> {
        DiffTensor::concat(tensors, 0, false)
    }

    fn hstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, TensorError> {
        DiffTensor::concat(tensors, 1, false)
    }

    fn dstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, TensorError> {
        DiffTensor::concat(tensors, 2, false)
    }
}
