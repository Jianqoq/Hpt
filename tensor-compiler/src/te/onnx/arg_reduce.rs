use std::panic::Location;
use std::sync::Arc;

use tensor_types::dtype::Dtype;

use crate::halide::alloca_stmt::AllocaStmt;
use crate::halide::exprs::{ Load, Lt };
use crate::halide::if_stmt::IfThenElse;
use crate::halide::let_stmt::LetStmt;
use crate::halide::primitive_type::PrimitiveType;
use crate::halide::seq_stmt::Seq;
use crate::halide::store_stmt::StoreStmt;
use crate::halide::utils::{ dtype_inf, dtype_neginf };
use crate::halide::variable::Variable;
use crate::iter_var::IterVar;
use crate::te::context::Context;
use crate::te::index_replace::reduce_replace;
use crate::te::stages::{ Body, ReduceStage };
use crate::halide::prime_expr::PrimeExpr;
use crate::te::stages::Stage;
use crate::halide::stmt::Stmt;
use crate::te::strides_cal_helper::reduce_strides_cal;
use crate::te::tensor::{ StridesCal, Tensor };
use crate::to_prim_expr::ToPrimeExpr;

pub fn arg_reduce(
    is_output: bool,
    inputs: &Vec<Body>,
    original_shape: &Vec<PrimeExpr>,
    axis: i64,
    init: PrimeExpr,
    max: bool,
    tmp_init: PrimeExpr,
    output_id: usize
) -> Body {
    let init = |stage: &Stage| {
        vec![
            Body::Stmt(
                Stmt::AllocaStmt(
                    AllocaStmt::make(
                        &Variable::make(&format!("%{}_idx_ptr", output_id)),
                        PrimitiveType::Dtype(Dtype::I64),
                        1i64,
                        Stmt::None
                    )
                ).into()
            ),
            Body::Stmt(
                Stmt::StoreStmt(
                    StoreStmt::make(
                        &Variable::make(&format!("%{}_idx_ptr", output_id)),
                        0i64,
                        init.clone()
                    )
                ).into()
            ),
            Body::Stmt(
                Stmt::AllocaStmt(
                    AllocaStmt::make(
                        &Variable::make(&format!("%{}_val_ptr", output_id)),
                        PrimitiveType::Dtype(stage.dtype),
                        1i64,
                        Stmt::None
                    )
                ).into()
            ),
            Body::Stmt(
                Stmt::StoreStmt(
                    StoreStmt::make(
                        &Variable::make(&format!("%{}_val_ptr", output_id)),
                        0i64,
                        tmp_init.clone()
                    )
                ).into()
            )
        ]
    };
    if is_output {
        if let Body::Stage(stage) = &inputs[0] {
            arg_reduce_helper(original_shape, axis, init, output_id, stage, max, |origin_dims|
                vec![
                    Body::Stmt(
                        LetStmt::make(
                            &Variable::make(&format!("%{}_val", output_id)),
                            Load::make(&format!("%{}_idx_ptr", output_id), 0),
                            false,
                            Stmt::None
                        ).into()
                    ),
                    Body::Stmt(
                        StoreStmt::make(
                            &Variable::make(&format!("%{}", output_id)),
                            origin_dims
                                .iter()
                                .enumerate()
                                .map(
                                    |(idx, x)|
                                        PrimeExpr::Variable(x.var().clone()) *
                                        Load::make(&format!("%{}.s", output_id), idx).into()
                                )
                                .reduce(|acc, x| acc + x)
                                .unwrap_or(0i64.into()),
                            Variable::make(&format!("%{}_val", output_id))
                        ).into()
                    )
                ]
            )
        } else {
            panic!("input is not a stage");
        }
    } else {
        if let Body::Stage(stage) = &inputs[0] {
            arg_reduce_helper(original_shape, axis, init, output_id, stage, max, |_|
                vec![
                    Body::Stmt(
                        LetStmt::make(
                            &Variable::make(&format!("%{}_val", output_id)),
                            Load::make(&format!("%{}_idx_ptr", output_id), 0),
                            false,
                            Stmt::None
                        ).into()
                    )
                ]
            )
        } else {
            panic!("input is not a stage");
        }
    }
}

pub fn arg_reduce_helper<F, D>(
    original_shape: &Vec<PrimeExpr>,
    axis: i64,
    init: D,
    output_id: usize,
    stage: &Stage,
    max: bool,
    posts: F
)
    -> Body
    where F: Fn(&Vec<IterVar>) -> Vec<Body>, D: Fn(&Stage) -> Vec<Body>
{
    let mut stage = stage.clone();
    let mut bodys = stage.bodys.clone();
    reduce_replace(original_shape.len(), &vec![axis as usize], &mut bodys, stage.id, output_id);
    let loade_val: PrimeExpr = Variable::make(&format!("%{}_val", output_id)).into();
    let stage_out: PrimeExpr = Variable::make(&format!("%{}_val", stage.out_id)).into();
    let val_ptr = Variable::make(&format!("%{}_val_ptr", output_id));
    let idx_ptr = Variable::make(&format!("%{}_idx_ptr", output_id));
    let cond = if max {
        Lt::make(&loade_val, &stage_out)
    } else {
        Lt::make(&stage_out, &loade_val)
    };
    let true_stmt = Seq::make(
        vec![
            StoreStmt::make(&val_ptr, 0, stage_out),
            StoreStmt::make(&idx_ptr, 0, Variable::make(&format!("{}red{}", output_id, axis)))
        ]
    );
    let if_then_else = IfThenElse::make(cond, true_stmt, Stmt::None);
    bodys.push(Body::Stmt(if_then_else.into()));
    stage.dims = (0..stage.dims.len())
        .filter(|x| *x != (axis as usize))
        .enumerate()
        .map(|(idx, x)| {
            IterVar::new(0i64, original_shape[x].clone(), 1i64, &format!("ax{}", idx))
        })
        .collect();
    let red_axes = vec![
        IterVar::new(
            0i64,
            original_shape[axis as usize].clone(),
            1i64,
            &format!("{}red{}", output_id, axis)
        )
    ];

    let mut res_bodys = vec![
        Body::Stmt(
            LetStmt::make(
                &Variable::make(&format!("%{}_val", output_id)),
                Load::make(&format!("%{}_val_ptr", output_id), 0),
                false,
                Stmt::None
            ).into()
        )
    ];
    res_bodys.extend(bodys);
    let reduce_stage = ReduceStage {
        dims: red_axes.clone(),
        bodys: res_bodys,
        id: output_id,
        inits: init(&stage),
        posts: posts(&stage.dims),
        input: stage.id,
    };
    stage.out_id = output_id;
    stage.bodys = vec![Body::ReduceStage(reduce_stage)];
    Body::Stage(stage)
}

impl Context {
    #[track_caller]
    pub fn argmin(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, mut axis: i64) -> Tensor {
        if axis < 0 {
            axis += a.shape.len() as i64;
        }

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if i != axis {
                res_shape.push(a.shape[i as usize].clone());
            }
        }
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let a_shape = a.shape.clone();
        let res_shape = Arc::new(res_shape);
        let init = init.to_prime_expr();
        let inf = dtype_inf(a.dtype);
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: Dtype::I64,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                reduce_strides_cal(prev_fn[0].clone(), vec![axis as usize])
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let inf = inf.clone();
                arg_reduce(is_output, &inputs, &a_shape, axis, init.clone(), false, inf, id)
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn argmax(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, mut axis: i64) -> Tensor {
        if axis < 0 {
            axis += a.shape.len() as i64;
        }

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if i != axis {
                res_shape.push(a.shape[i as usize].clone());
            }
        }
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let a_shape = a.shape.clone();
        let res_shape = Arc::new(res_shape);
        let init = init.to_prime_expr();
        let neginf = dtype_neginf(a.dtype);
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: Dtype::I64,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                reduce_strides_cal(prev_fn[0].clone(), vec![axis as usize])
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let neginf = neginf.clone();
                arg_reduce(is_output, &inputs, &a_shape, axis, init.clone(), true, neginf, id)
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
