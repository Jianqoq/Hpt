use std::{ panic::Location, sync::Arc };

use crate::{
    halide::{ prime_expr::PrimeExpr, store_stmt::StoreStmt, tensor_load::Flag, utils::var },
    iter_var::IterVar,
    te::{
        context::Context,
        stages::{ Body, Stage, Switch },
        strides_cal_helper::elementwise_strides_cal,
        tensor::{ StridesCal, Tensor },
    },
};
use crate::halide::exprs::Load;
use crate::halide::let_stmt::LetStmt;
use crate::halide::stmt::Stmt;
use crate::halide::tensor_load::TensorLoad;

impl Context {
    #[track_caller]
    pub fn concat(&mut self, axis: usize, inputs: &[Tensor]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let concat_dim = inputs
            .iter()
            .map(|x| x.shape[axis].clone())
            .reduce(|x, y| x + y)
            .unwrap();
        let mut res_shape = inputs[0].shape.as_ref().clone();
        res_shape[axis] = concat_dim;
        let res_shape = Arc::new(res_shape);
        let res_dtype = inputs[0].dtype.clone();
        let len = inputs.len();
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(
                inputs
                    .iter()
                    .map(|x| x.id)
                    .collect()
            ),
            span: Location::caller(),
            id,
            dtype: res_dtype,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                elementwise_strides_cal(prev_fn[0].clone())
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let shape = res_shape.clone();
                let dims = vec![IterVar::new(0i64, len as i64, 1i64, var("idx"))];
                let mut cases = vec![];
                for (case, i) in inputs.iter().enumerate() {
                    let body = match i {
                        Body::Stage(stage) => {
                            let mut stage = stage.clone();
                            let mut begins = vec![];
                            for i in 0..shape.len() {
                                if i == axis {
                                    begins.push((case as i64).into());
                                } else {
                                    begins.push((0i64).into());
                                }
                            }
                            let mut axes = vec![];
                            let mut cnt = 0;
                            for i in 0..shape.len() {
                                if i == axis {
                                    axes.push(var("idx").into());
                                } else {
                                    axes.push(stage.dims[cnt].var().into());
                                    cnt += 1;
                                }
                            }
                            let mut steps = vec![];
                            for _ in 0..shape.len() {
                                steps.push((1i64).into());
                            }
                            let mut strides = vec![];
                            for i in 0..shape.len() {
                                strides.push(Load::make(var(format!("{}.s", id)), i as i64).into());
                            }
                            stage.bodys.push(
                                Body::Stmt(
                                    StoreStmt::make(
                                        var(format!("%{}", id)),
                                        begins,
                                        axes,
                                        steps,
                                        strides,
                                        var(format!("%{}_val", stage.out_id))
                                    ).into()
                                )
                            );
                            Body::Stage(stage)
                        }
                        Body::Switch(_) => todo!(),
                        _ => { i.clone() }
                    };
                    cases.push(((case as i64).into(), body));
                }
                if is_output {
                    let switch = Switch {
                        dims: dims.clone(),
                        cond: var("idx").into(),
                        dtype: res_dtype,
                        bodys: cases,
                        id,
                        input: id,
                        out_id: id,
                    };
                    Body::Switch(switch)
                } else {
                    let switch = Switch {
                        dims: dims.clone(),
                        cond: var("idx").into(),
                        bodys: cases,
                        dtype: res_dtype,
                        id,
                        input: id,
                        out_id: id,
                    };
                    Body::Switch(switch)
                }
            }),
        };
        self.nodes.borrow_mut().insert(ret.id, ret.clone());
        ret
    }
}
