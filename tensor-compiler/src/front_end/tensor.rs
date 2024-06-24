#![allow(dead_code)]
use std::{ cell::RefCell, rc::Rc };

use hashbrown::HashSet;
use tensor_common::{
    axis::{ process_axes, Axis },
    err_handler::ErrHandler,
    layout::Layout,
    shape::Shape,
    strides::Strides,
};
use tensor_traits::tensor::{ CommonBounds, StaticTensorInfo };
use tensor_types::{ convertion::Convertor, dtype::Dtype };

use crate::{ float::Float, op::Op };

use super::context::_Context;

pub struct Tensor {
    pub(crate) inputs: Rc<Vec<usize>>,
    pub(crate) dtype: Dtype,
    pub(crate) op: Op,
    pub(crate) layout: Layout,
    pub(crate) name: Option<Rc<String>>,
    pub(crate) error_msg: Rc<Vec<ErrHandler>>,
    pub(crate) block_id: usize,
    pub(crate) id: usize,
    pub(crate) ctx: Rc<RefCell<_Context>>,
}

impl StaticTensorInfo for Tensor {
    fn size(&self) -> usize {
        self.layout.size()
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl Tensor {
    pub(crate) fn _empty(
        inputs: Vec<usize>,
        dtype: Dtype,
        op: Op,
        layout: Layout,
        name: Option<Rc<String>>,
        error_msg: Rc<Vec<ErrHandler>>,
        ctx: Rc<RefCell<_Context>>
    ) -> Tensor {
        let block_id = ctx.borrow().block_stack().last().unwrap().current_id();
        let ret = Tensor {
            inputs: inputs.into(),
            dtype,
            op,
            layout,
            name,
            error_msg,
            block_id,
            id: *ctx.borrow().acc_node_id(),
            ctx: ctx.clone(),
        };
        ctx.borrow_mut().increment_id();
        ret
    }

    pub(crate) fn arange<T: Convertor + CommonBounds>(
        start: T,
        end: T,
        step: T,
        ctx: Rc<RefCell<_Context>>
    ) -> Tensor {
        let start = start.to_f64();
        let end = end.to_f64();
        let step = step.to_f64();
        let shape: Shape = vec![((end - start) / step) as i64].into();
        let strides = shape.to_strides();
        Tensor::_empty(
            vec![],
            T::ID,
            Op::Arange {
                start: Float::new(start).into(),
                end: Float::new(end).into(),
                step: Float::new(step).into(),
            },
            Layout::new(shape, strides),
            None,
            Rc::new(vec![]),
            ctx
        )
    }

    pub(crate) fn randn<S: Into<Shape>>(
        ctx: Rc<RefCell<_Context>>,
        mean: f64,
        std: f64,
        shape: S,
        dtype: Dtype
    ) -> Self {
        let shape = shape.into();
        let strides = shape.to_strides();
        let op = Op::Randn {
            mean: Float::new(mean).into(),
            std: Float::new(std).into(),
        };
        Self::_empty(vec![], dtype, op, Layout::new(shape, strides), None, Rc::new(vec![]), ctx)
    }

    pub(crate) fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
        ctx: Rc<RefCell<_Context>>
    ) -> Self {
        let axes: Vec<usize> = process_axes(axes, self.ndim()).unwrap();
        let shape: Shape = self
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(idx, &x)| if axes.contains(&idx) { None } else { Some(x) })
            .collect::<Vec<i64>>()
            .into();
        let strides = shape.to_strides();
        let layout = Layout::new(shape, strides);
        let ret = Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Null,
            layout,
            None,
            Rc::new(vec![]),
            ctx.clone()
        );
        let mut ret = if keep_dims {
            let mut res_shape = self.shape().to_vec();
            for i in axes.iter() {
                res_shape[*i] = 1;
            }
            ret.reshape(res_shape)
        } else {
            ret
        };
        ret.op = Op::Sum { axes: axes.into() };
        ret
    }

    pub(crate) fn reshape<S: Into<Shape>>(&self, shape: S) -> Self {
        let res_shape = shape.into();
        let err = ErrHandler::check_size_match(self.shape(), &res_shape);
        if let Err(err) = err {
            todo!("{:?}", err);
        }
        if let Some(new_strides) = self.layout.is_reshape_possible(&res_shape) {
            let ret = Tensor::_empty(
                vec![self.id],
                self.dtype,
                Op::Reshape,
                Layout::new(res_shape.clone(), new_strides),
                None,
                Rc::new(vec![]),
                self.ctx.clone()
            );
            ret
        } else {
            return self.contiguous().reshape(res_shape);
        }
    }

    pub(crate) fn contiguous(&self) -> Self {
        let strides = if !self.layout.is_contiguous() {
            self.shape().to_strides()
        } else {
            self.strides().clone()
        };
        let ret = Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Contiguous,
            Layout::new(self.shape().clone(), strides),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        );
        ret
    }

    pub fn permute<A: Into<Axis>>(&self, axes: A) -> Self {
        let axes = process_axes(axes, self.ndim()).unwrap();
        if axes.len() != (self.shape().len() as usize) {
            panic!(
                "Axes length mismatch in permute method, expected: {}, got: {}",
                self.shape().len(),
                axes.len()
            );
        }
        let mut set = HashSet::new();
        for i in axes.iter().enumerate() {
            if *i.1 >= (self.shape().len() as usize) {
                panic!(
                    "Axes {} out of range(Should be {}..{}). Pos: {}",
                    i.1,
                    0,
                    self.shape().len(),
                    i.0
                );
            }
            if !set.insert(*i.1) {
                panic!("Axes {} repeated. Pos: {}", i.1, i.0);
            }
        }
        let mut new_shape = self.shape().to_vec();
        for i in axes.iter() {
            new_shape[*i] = self.shape()[axes[*i]];
        }
        let mut new_strides = self.strides().to_vec();
        for i in axes.iter() {
            new_strides[*i] = self.strides()[axes[*i]];
        }
        let ret = Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Transpose { axes: axes.into() },
            Layout::new(new_shape.into(), new_strides.into()),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        );
        ret
    }

    pub fn abs(&self) -> Self {
        Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Abs,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        )
    }
}
