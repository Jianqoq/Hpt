use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{ store_stmt::StoreStmt, utils::var },
    iter_var::IterVar,
    te::{ context::Context, stages::{ Body, Switch }, tensor::{ StridesCal, Tensor } },
};
use crate::halide::exprs::Load;

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
                {
                    let prev_funcs = prev_fn
                        .iter()
                        .map(|x| x.clone())
                        .collect::<Vec<_>>();
                    Arc::new(move |map: &HashMap<Arc<String>, i64>| {
                        let mut ret = vec![];
                        for func in prev_funcs.iter() {
                            ret.append(&mut func(map));
                        }
                        ret
                    })
                }
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let shape = res_shape.clone();
                let dims = vec![IterVar::new(0i64, len as i64, 1i64, var("idx"))];
                if is_output {
                    let mut cases = vec![];
                    let mut offsets = vec![];
                    for (case, i) in inputs.iter().enumerate() {
                        let body = match i {
                            Body::Stage(stage) => {
                                let mut stage = stage.clone();
                                let mut begins = vec![];
                                for i in 0..shape.len() {
                                    if i == axis {
                                        let offset = offsets[..case]
                                            .iter()
                                            .cloned()
                                            .reduce(|x, y| x + y)
                                            .unwrap_or((0i64).into());
                                        begins.push(offset);
                                    } else {
                                        begins.push((0i64).into());
                                    }
                                }
                                offsets.push(stage.dims[axis].end().clone());
                                let mut axes = vec![];
                                for i in 0..shape.len() {
                                    axes.push(stage.dims[i].var().into());
                                }
                                let mut steps = vec![];
                                for _ in 0..shape.len() {
                                    steps.push((1i64).into());
                                }
                                let mut strides = vec![];
                                for i in 0..shape.len() {
                                    strides.push(
                                        Load::make(var(format!("{}.s", id)), i as i64).into()
                                    );
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
                    let mut cases = vec![];
                    for (case, i) in inputs.iter().enumerate() {
                        cases.push(((case as i64).into(), i.clone()));
                    }
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
