#![allow(dead_code)]
use std::{ cell::RefCell, rc::Rc };
use hashbrown::HashSet;
use tensor_common::{
    axis::{ process_axes, Axis },
    err_handler::ErrHandler,
    layout::Layout,
    shape::Shape,
    shape_utils::yield_one_after,
    strides::Strides,
};
use tensor_traits::tensor::{ CommonBounds, StaticTensorInfo };
use tensor_types::{ convertion::Convertor, dtype::Dtype, type_promote::{ FloatOut, NormalOut } };

use crate::{ float::Float, op::Op };

use super::context::_Context;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) inputs: Rc<Vec<usize>>,
    pub(crate) dtype: Dtype,
    pub(crate) op: Op,
    pub(crate) const_val: Option<Float>,
    pub(crate) layout: Layout,
    pub(crate) name: Option<Rc<String>>,
    pub(crate) error_msg: Rc<Vec<ErrHandler>>,
    pub(crate) block_id: usize,
    pub(crate) id: usize,
    pub(crate) ctx: Rc<RefCell<_Context>>,
}

macro_rules! impl_trigs {
    ($fn_name:ident, $op:ident, $method:ident) => {
        pub fn $fn_name(&self) -> Tensor {
            Tensor::_empty(
                vec![self.id],
                self.dtype.$method(),
                Op::$op,
                self.layout.clone(),
                None,
                Rc::new(vec![]),
                self.ctx.clone()
            )
        }
    };
}

macro_rules! impl_reduce {
    (
        $fn_name:ident,
        $op_name:ident,
        $($method:tt)*
    ) => {
        pub fn $fn_name<A: Into<Axis>>(&self, axes: A, keep_dims: bool) -> Self {
            let axes = process_axes(axes, self.ndim()).unwrap();
            let shape: Shape = self
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(idx, &x)| if axes.contains(&idx) { None } else { Some(x) })
                .collect::<Vec<i64>>()
                .into();
            let strides = shape.to_strides();
            let ret = Tensor::_empty(
                vec![self.id],
                self.dtype$($method)*,
                Op::$op_name { axes: axes.clone().into() },
                Layout::new(shape, strides),
                None,
                Rc::new(vec![]),
                self.ctx.clone()
            );
            if keep_dims {
                let mut res_shape = self.shape().to_vec();
                for i in axes.iter() {
                    res_shape[*i] = 1;
                }
                ret.reshape(res_shape)
            } else {
                ret
            }
        }
    };
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
    impl_reduce!(var, Var, ._div(Dtype::Usize));
    impl_reduce!(mean, Mean, ._div(Dtype::Usize));
    impl_reduce!(max, Max, ); // prettier-ignore
    impl_reduce!(min, Min, ); // prettier-ignore
    impl_reduce!(sum, Sum, ); // prettier-ignore

    impl_trigs!(sin, Sin, _sin);
    impl_trigs!(cos, Cos, _cos);
    impl_trigs!(tan, Tan, _tan);
    impl_trigs!(asin, Asin, _asin);
    impl_trigs!(acos, Acos, _acos);
    impl_trigs!(atan, Atan, _atan);
    impl_trigs!(sinh, Sinh, _sinh);
    impl_trigs!(cosh, Cosh, _cosh);
    impl_trigs!(tanh, Tanh, _tanh);
    impl_trigs!(asinh, Asinh, _asinh);
    impl_trigs!(acosh, Acosh, _acosh);
    impl_trigs!(atanh, Atanh, _atanh);

    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string().into());
    }

    pub(crate) fn scalar<T: Convertor + CommonBounds>(
        ctx: Rc<RefCell<_Context>>,
        scalar: T
    ) -> Self {
        let mut ret = Tensor::_empty(
            vec![],
            T::ID,
            Op::Null,
            Layout::new(vec![1].into(), vec![1].into()),
            None,
            Rc::new(vec![]),
            ctx
        );
        ret.const_val = Some(Float::new(scalar.to_f64()));
        ret
    }

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
            const_val: None,
            layout,
            name,
            error_msg,
            block_id,
            id: *ctx.borrow().acc_node_id(),
            ctx: ctx.clone(),
        };
        ctx.borrow_mut().nodes_mut().insert(ret.id, ret.clone().into());
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
        let ret = Self::_empty(
            vec![],
            dtype,
            op,
            Layout::new(shape, strides),
            None,
            Rc::new(vec![]),
            ctx
        );
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
            self.dtype._abs(),
            Op::Abs,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        )
    }

    pub fn pow(&self, power: Tensor) -> Self {
        let ret = Tensor::_empty(
            vec![self.id, power.id],
            self.dtype._pow(power.dtype),
            Op::Power,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        );
        ret
    }

    pub fn square(&self) -> Self {
        let ret = Tensor::_empty(
            vec![self.id],
            self.dtype._square(),
            Op::Square,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        );
        ret
    }

    pub fn sqrt(&self) -> Self {
        let ret = Tensor::_empty(
            vec![self.id],
            self.dtype._sqrt(),
            Op::Sqrt,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        );
        ret
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Self {
        let res_shape: Shape = shape.into().into();
        let mut new_strides = self.strides().to_vec();
        self.shape()
            .iter()
            .enumerate()
            .rev()
            .zip(res_shape.iter().rev())
            .for_each(|((idx, x), y)| {
                if x != y {
                    if x == &1 {
                        new_strides[idx] = 0;
                    } else {
                        panic!(
                            "input shape {:?} can not be expanded to {:?}, please make sure input shape[{}]: {} equals 1 if you want to expand",
                            self.shape(),
                            res_shape,
                            idx,
                            self.shape()[idx]
                        )
                    }
                }
            });
        let mut res_strides = vec![0; res_shape.len()];
        res_strides
            .iter_mut()
            .rev()
            .zip(new_strides.iter().rev())
            .for_each(|(x, y)| {
                *x = *y;
            });
        let ret = Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Expand,
            Layout::new(res_shape, res_strides.into()),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        );
        ret
    }

    pub fn repeat(&self, repeats: usize, axes: i16) -> Self {
        let mut val: usize = axes as usize;
        if axes < 0 {
            val = self.shape().len() + (axes as usize);
        }
        let mut new_shape = yield_one_after(&self.shape(), val);
        let mut new_tensor = self.reshape(&new_shape);
        new_shape[val + 1] *= repeats as i64;
        new_tensor = new_tensor.expand(new_shape);
        new_shape = self.shape().to_vec();
        new_shape[val] *= repeats as i64;
        new_tensor.reshape(new_shape)
    }

    pub fn floor(&self) -> Self {
        Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Floor,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        )
    }

    pub fn ceil(&self) -> Self {
        Tensor::_empty(
            vec![self.id],
            self.dtype,
            Op::Ceil,
            self.layout.clone(),
            None,
            Rc::new(vec![]),
            self.ctx.clone()
        )
    }
}
