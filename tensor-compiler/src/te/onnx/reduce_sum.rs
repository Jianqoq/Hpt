use std::{ panic::Location, sync::Arc };

use crate::{
    halide::exprs::{ Max, Min },
    te::{
        bodygen_helper::common_reduce,
        context::Context,
        stages::Body,
        strides_cal_helper::reduce_strides_cal,
        tensor::{ StridesCal, Tensor },
    },
    to_prim_expr::ToPrimeExpr,
};

impl Context {
    #[track_caller]
    pub fn sum(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, axes: &[i64]) -> Tensor {
        let mut axes = axes.to_vec();
        for i in axes.iter_mut() {
            if *i < 0 {
                *i += a.shape.len() as i64;
            }
        }
        axes.sort();

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if !axes.contains(&i) {
                res_shape.push(a.shape[i as usize].clone());
            }
        }
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let a_shape = a.shape.clone();
        let res_shape = Arc::new(res_shape);
        let init = init.to_prime_expr();
        let axes = Arc::new(axes);
        let axes1 = axes.clone();
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: a.dtype.clone(),
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                reduce_strides_cal(
                    prev_fn[0].clone(),
                    axes1
                        .iter()
                        .map(|x| *x as usize)
                        .collect()
                )
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let axes = axes
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<usize>>();
                common_reduce(is_output, &inputs, &a_shape, &axes, init.clone(), |x, y| x + y, id)
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn prod(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, axes: &[i64]) -> Tensor {
        let mut axes = axes.to_vec();
        for i in axes.iter_mut() {
            if *i < 0 {
                *i += a.shape.len() as i64;
            }
        }
        axes.sort();

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if !axes.contains(&i) {
                res_shape.push(a.shape[i as usize].clone());
            }
        }
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let a_shape = a.shape.clone();
        let res_shape = Arc::new(res_shape);
        let init = init.to_prime_expr();
        let axes = Arc::new(axes);
        let axes1 = axes.clone();
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: a.dtype.clone(),
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                reduce_strides_cal(
                    prev_fn[0].clone(),
                    axes1
                        .iter()
                        .map(|x| *x as usize)
                        .collect()
                )
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let axes = axes
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<usize>>();
                common_reduce(is_output, &inputs, &a_shape, &axes, init.clone(), |x, y| x * y, id)
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn max(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, axes: &[i64]) -> Tensor {
        let mut axes = axes.to_vec();
        for i in axes.iter_mut() {
            if *i < 0 {
                *i += a.shape.len() as i64;
            }
        }
        axes.sort();

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if !axes.contains(&i) {
                res_shape.push(a.shape[i as usize].clone());
            }
        }
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let a_shape = a.shape.clone();
        let res_shape = Arc::new(res_shape);
        let init = init.to_prime_expr();
        let axes = Arc::new(axes);
        let axes1 = axes.clone();
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: a.dtype.clone(),
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                reduce_strides_cal(
                    prev_fn[0].clone(),
                    axes1
                        .iter()
                        .map(|x| *x as usize)
                        .collect()
                )
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let axes = axes
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<usize>>();
                common_reduce(
                    is_output,
                    &inputs,
                    &a_shape,
                    &axes,
                    init.clone(),
                    |x, y| Max::make(x, y).into(),
                    id
                )
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn min(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, axes: &[i64]) -> Tensor {
        let mut axes = axes.to_vec();
        for i in axes.iter_mut() {
            if *i < 0 {
                *i += a.shape.len() as i64;
            }
        }
        axes.sort();

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if !axes.contains(&i) {
                res_shape.push(a.shape[i as usize].clone());
            }
        }
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let a_shape = a.shape.clone();
        let res_shape = Arc::new(res_shape);
        let init = init.to_prime_expr();
        let axes = Arc::new(axes);
        let axes1 = axes.clone();
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: a.dtype.clone(),
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                reduce_strides_cal(
                    prev_fn[0].clone(),
                    axes1
                        .iter()
                        .map(|x| *x as usize)
                        .collect()
                )
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let axes = axes
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<usize>>();
                common_reduce(
                    is_output,
                    &inputs,
                    &a_shape,
                    &axes,
                    init.clone(),
                    |x, y| Min::make(x, y).into(),
                    id
                )
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
