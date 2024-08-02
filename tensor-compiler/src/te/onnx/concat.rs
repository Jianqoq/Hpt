use std::{panic::Location, sync::Arc};

use crate::{
    halide::{store_stmt::StoreStmt, utils::var},
    iter_var::IterVar,
    te::{
        context::Context,
        stages::{Body, Stage},
        strides_cal_helper::elementwise_strides_cal,
        tensor::{StridesCal, Tensor},
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
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(inputs.iter().map(|x| x.id).collect()),
            span: Location::caller(),
            id,
            dtype: res_dtype,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                elementwise_strides_cal(prev_fn[0].clone())
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let shape = res_shape.clone();
                let mut dims = vec![IterVar::new(
                    0i64,
                    shape[axis].clone(),
                    1i64,
                    var("ax0".to_string()),
                )];
                let mut cnt = 1;
                for (idx, i) in shape.iter().enumerate() {
                    if idx != axis {
                        dims.push(IterVar::new(
                            0i64,
                            i.clone(),
                            1i64,
                            var(format!("ax{}", cnt)),
                        ));
                        cnt += 1;
                    }
                }
                if is_output {
                    let load = Load::make(var("data_vec".to_string()), var("ax0".to_string()));
                    let let_stmt = Body::Stmt(
                        LetStmt::make(&var(format!("%{}cat_ptr", id)), load, false, Stmt::None)
                            .into(),
                    );
                    let begins = (0..shape.len()).map(|(_)| 0i64.into()).collect();
                    let steps = (0..shape.len()).map(|(_)| 1i64.into()).collect();
                    let axes = (0..shape.len())
                        .map(|idx| var(format!("ax{}", idx)).into())
                        .collect();
                    let strides = (0..shape.len())
                        .map(|idx| Load::make(var(format!("{}.s", id)), idx))
                        .collect();
                    let tensor_load = TensorLoad::make(
                        var(format!("%{}cat_ptr", id)),
                        begins,
                        axes,
                        steps,
                        strides,
                        vec![],
                    );
                    let body = Body::Stmt(
                        StoreStmt::make(
                            var(format!("%{}", id)),
                            (0..shape.len()).map(|_| 0i64.into()).collect(),
                            (0..shape.len())
                                .map(|idx| var(format!("ax{}", idx)))
                                .collect(),
                            (0..shape.len()).map(|_| 1i64).collect(),
                            (0..shape.len())
                                .map(|idx| Load::make(var(format!("{}.s", id)), idx))
                                .collect(),
                            tensor_load,
                        )
                        .into(),
                    );
                    let ret = Stage {
                        dims,
                        bodys: vec![let_stmt, body],
                        dtype: res_dtype,
                        id,
                        out_id: id,
                        begins: (0..shape.len()).map(|_| 0i64.into()).collect(),
                        steps: (0..shape.len()).map(|_| 1i64.into()).collect(),
                        axes: (0..shape.len())
                            .map(|idx| var(format!("ax{}", idx)).into())
                            .collect(),
                    };
                    return Body::Stage(ret);
                } else {
                    let load = Load::make(var("data_vec".to_string()), var("ax0".to_string()));
                    let let_stmt = Body::Stmt(
                        LetStmt::make(&var(format!("%{}cat_ptr", id)), load, false, Stmt::None)
                            .into(),
                    );
                    let begins = (0..shape.len()).map(|(_)| 0i64.into()).collect();
                    let steps = (0..shape.len()).map(|(_)| 1i64.into()).collect();
                    let axes = (0..shape.len())
                        .map(|idx| var(format!("ax{}", idx)).into())
                        .collect();
                    let strides = (0..shape.len())
                        .map(|idx| Load::make(var(format!("{}.s", id)), idx))
                        .collect();
                    let tensor_load = TensorLoad::make(
                        var(format!("%{}cat_ptr", id)),
                        begins,
                        axes,
                        steps,
                        strides,
                        vec![],
                    );
                    let body = Body::Stmt(
                        LetStmt::make(&var(format!("%{}_val", id)), tensor_load, false, Stmt::None)
                            .into(),
                    );
                    let ret = Stage {
                        dims,
                        bodys: vec![let_stmt, body],
                        dtype: res_dtype,
                        id,
                        out_id: id,
                        begins: (0..shape.len()).map(|_| 0i64.into()).collect(),
                        steps: (0..shape.len()).map(|_| 1i64.into()).collect(),
                        axes: (0..shape.len())
                            .map(|idx| var(format!("ax{}", idx)).into())
                            .collect(),
                    };
                    return Body::Stage(ret);
                }
            }),
        };
        self.nodes.borrow_mut().insert(ret.id, ret.clone());
        ret
    }
}
