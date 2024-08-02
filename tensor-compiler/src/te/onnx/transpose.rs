use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        store_stmt::StoreStmt,
        substitute::subsititue_var::SubstituteVar,
        utils::var,
    },
    iter_var::IterVar,
    te::{
        context::Context,
        hstrides::HStrides,
        stages::{ Body, Stage },
        tensor::{ StridesCal, Tensor },
    },
};

impl Context {
    #[track_caller]
    pub fn transpose(&mut self, a: &Tensor, axes: &[i64]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let shape = a.shape.clone();
        let dtype = a.dtype.clone();
        let mut axes = axes.to_vec();
        for i in axes.iter_mut() {
            if *i < 0 {
                *i += shape.len() as i64;
            }
        }
        let new_shape = Arc::new(
            axes
                .iter()
                .map(|&i| shape[i as usize].clone())
                .collect::<Vec<PrimeExpr>>()
        );
        let ret = Tensor {
            shape: new_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                {
                    let prev_func = prev_fn[0].clone();
                    let axes = axes.clone();
                    Arc::new(move |map: &HashMap<Arc<String>, i64>| {
                        let strides = prev_func(map);
                        let trasponsed_strides = strides
                            .iter()
                            .map(|strides| {
                                let mut new_strides = vec![];
                                for &i in axes.iter() {
                                    new_strides.push(strides.strides[i as usize]);
                                }
                                HStrides {
                                    strides: new_strides,
                                    reduced_dim: strides.reduced_dim,
                                    offset: strides.offset,
                                }
                            })
                            .collect::<Vec<HStrides>>();
                        trasponsed_strides
                    })
                }
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let new_shape = new_shape.clone();
                if is_output {
                    if let Body::Stage(stage) = &inputs[0] {
                        let mut stage = stage.clone();
                        let dims = (0..new_shape.len())
                            .map(|x|
                                IterVar::new(0i64, new_shape[x].clone(), 1i64, &format!("ax{}", x))
                            )
                            .collect::<Vec<IterVar>>();
                        let mut subs_var = SubstituteVar::new();
                        subs_var.add_replacement(
                            var(format!("%{}.s", stage.out_id)),
                            var(format!("%{}.s", id))
                        );
                        stage.bodys.push(
                            Body::Stmt(
                                StoreStmt::make(
                                    var(format!("%{}", id)),
                                    stage.begins.clone(),
                                    stage.axes.clone(),
                                    stage.steps.clone(),
                                    (0..dims.len())
                                        .map(|x| { Load::make(&format!("%{}.s", id), x).into() })
                                        .collect::<Vec<PrimeExpr>>(),
                                    var(&format!("%{}_val", stage.out_id))
                                ).into()
                            )
                        );
                        let stage = Stage {
                            dims,
                            bodys: stage.bodys.clone(),
                            id,
                            out_id: id,
                            dtype,
                            begins: stage.begins.clone(),
                            steps: stage.steps.clone(),
                            axes: stage.axes.clone(),
                        };
                        Body::Stage(stage)
                    } else {
                        panic!("Transpose body_gen: inputs[0] is not a stage");
                    }
                } else {
                    if let Body::Stage(stage) = &inputs[0] {
                        let mut stage = stage.clone();
                        let dims = (0..new_shape.len())
                            .map(|x|
                                IterVar::new(0i64, new_shape[x].clone(), 1i64, &format!("ax{}", x))
                            )
                            .collect::<Vec<IterVar>>();
                        let mut subs_var = SubstituteVar::new();
                        subs_var.add_replacement(
                            var(format!("%{}.s", stage.out_id)),
                            var(format!("%{}.s", id))
                        );
                        stage.bodys.push(
                            Body::Stmt(
                                Stmt::LetStmt(
                                    LetStmt::make(
                                        &var(&format!("%{}_val", id)),
                                        var(&format!("%{}_val", stage.out_id)),
                                        false,
                                        Stmt::None
                                    )
                                ).into()
                            )
                        );
                        let stage = Stage {
                            dims,
                            bodys: stage.bodys.clone(),
                            id,
                            out_id: id,
                            dtype,
                            begins: stage.begins.clone(),
                            steps: stage.steps.clone(),
                            axes: stage.axes.clone(),
                        };
                        Body::Stage(stage)
                    } else {
                        panic!("Transpose body_gen: inputs[0] is not a stage");
                    }
                }
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
