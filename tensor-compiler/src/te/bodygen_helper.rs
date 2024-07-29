use std::sync::Arc;

use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        alloca_stmt::AllocaStmt,
        exprs::{ Add, Call, Load },
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        primitive_type::PrimitiveType,
        stmt::Stmt,
        store_stmt::StoreStmt,
        substitute::subsititue_var::SubstituteVar,
        traits::MutatorGetSet,
        variable::Variable,
    },
    iter_var::IterVar,
    te::insert_axes::InsertAxes,
    to_prim_expr::ToPrimeExpr,
};

use super::{ index_replace::reduce_replace, stages::{ Body, ReduceStage, Stage } };

pub fn common_reduce(
    is_output: bool,
    inputs: &Vec<Body>,
    original_shape: &Vec<PrimeExpr>,
    axes: &Vec<usize>,
    init: PrimeExpr,
    output_id: usize
) -> Body {
    if is_output {
        if let Body::Stage(stage) = &inputs[0] {
            common_reduce_helper(original_shape, axes, init, output_id, stage, |origin_dims|
                common_reduce_out(origin_dims, output_id)
            )
        } else {
            panic!("input is not a stage");
        }
    } else {
        if let Body::Stage(stage) = &inputs[0] {
            common_reduce_helper(original_shape, axes, init, output_id, stage, |_|
                vec![
                    Body::Stmt(
                        Stmt::LetStmt(
                            LetStmt::make(
                                &Variable::make(&format!("%{}_val", output_id)),
                                Load::make(&format!("%{}_val_ptr", output_id), 0),
                                false,
                                Stmt::None
                            )
                        )
                    )
                ]
            )
        } else {
            panic!("input is not a stage");
        }
    }
}

pub fn common_reduce_helper<F>(
    original_shape: &Vec<PrimeExpr>,
    axes: &Vec<usize>,
    init: PrimeExpr,
    output_id: usize,
    stage: &Stage,
    posts: F
) -> Body
    where F: Fn(&Vec<IterVar>) -> Vec<Body>
{
    let mut stage = stage.clone();
    let mut bodys = stage.bodys.clone();
    reduce_replace(original_shape.len(), &axes, &mut bodys, stage.id, output_id);
    bodys.push(
        Body::Stmt(
            Stmt::StoreStmt(
                StoreStmt::make(
                    &Variable::make(&format!("%{}_val_ptr", output_id)),
                    0,
                    Add::make(
                        &Variable::make(&format!("%{}_val", output_id)),
                        Variable::make(&format!("%{}_val", stage.out_id))
                    )
                )
            )
        )
    );
    stage.dims = (0..stage.dims.len())
        .filter(|x| !axes.contains(x))
        .enumerate()
        .map(|(idx, x)| IterVar::new(0i64, original_shape[x].clone(), 1i64, &format!("ax{}", idx)))
        .collect();
    let red_axes = axes
        .iter()
        .map(|x|
            IterVar::new(0i64, original_shape[*x].clone(), 1i64, &format!("{}red{}", output_id, x))
        )
        .collect::<Vec<IterVar>>();

    let mut res_bodys = vec![
        Body::Stmt(
            Stmt::LetStmt(
                LetStmt::make(
                    &Variable::make(&format!("%{}_val", output_id)),
                    Load::make(&format!("%{}_val_ptr", output_id), 0),
                    false,
                    Stmt::None
                )
            )
        )
    ];
    res_bodys.extend(bodys);
    let reduce_stage = ReduceStage {
        dims: red_axes.clone(),
        bodys: res_bodys,
        id: output_id,
        inits: vec![
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
                    StoreStmt::make(&Variable::make(&format!("%{}_val_ptr", output_id)), 0i64, init)
                ).into()
            )
        ],
        posts: posts(&stage.dims),
        input: stage.id,
    };
    stage.out_id = output_id;
    stage.bodys = vec![Body::ReduceStage(reduce_stage)];
    Body::Stage(stage)
}

pub fn common_reduce_out(all_parent_dims: &Vec<IterVar>, output_id: usize) -> Vec<Body> {
    vec![
        Body::Stmt(
            Stmt::LetStmt(
                LetStmt::make(
                    &Variable::make(&format!("%{}_val", output_id)),
                    Load::make(&format!("%{}_val_ptr", output_id), 0),
                    false,
                    Stmt::None
                )
            )
        ),
        Body::Stmt(
            Stmt::StoreStmt(
                StoreStmt::make(
                    &Variable::make(&format!("%{}", output_id)),
                    all_parent_dims
                        .iter()
                        .enumerate()
                        .map(
                            |(idx, x)|
                                PrimeExpr::Variable(x.var().clone()) *
                                Load::make(&format!("%{}.s", output_id), idx).into()
                        )
                        .reduce(|acc, x| acc + x)
                        .unwrap_or((0i64).into()),
                    Variable::make(&format!("%{}_val", output_id))
                )
            )
        )
    ]
}

pub fn common_binop<F>(
    is_output: bool,
    inputs: &Vec<Body>,
    lhs_new_axes: &[usize],
    lhs_replace: &[(usize, usize)],
    rhs_new_axes: &[usize],
    rhs_replace: &[(usize, usize)],
    ty_infer: fn(Dtype, Dtype) -> Dtype,
    binop: F,
    dims: &Vec<IterVar>,
    output_id: usize
) -> Body
    where F: Fn(PrimeExpr, PrimeExpr) -> PrimeExpr + Clone
{
    if is_output {
        common_binop_helper(
            inputs,
            lhs_new_axes,
            lhs_replace,
            rhs_new_axes,
            rhs_replace,
            dims,
            output_id,
            ty_infer,
            || common_binop_out(dims, inputs[0].id(), inputs[1].id(), output_id, binop.clone())
        )
    } else {
        common_binop_helper(
            inputs,
            lhs_new_axes,
            lhs_replace,
            rhs_new_axes,
            rhs_replace,
            dims,
            output_id,
            ty_infer,
            || common_binop_in(inputs[0].id(), inputs[1].id(), binop.clone(), output_id)
        )
    }
}

pub fn common_binop_helper<F>(
    inputs: &Vec<Body>,
    lhs_new_axes: &[usize],
    lhs_replace: &[(usize, usize)],
    rhs_new_axes: &[usize],
    rhs_replace: &[(usize, usize)],
    dims: &Vec<IterVar>,
    output_id: usize,
    ty_infer: fn(Dtype, Dtype) -> Dtype,
    post: F
) -> Body
    where F: Fn() -> Body
{
    match (&inputs[0], &inputs[1]) {
        (Body::Stage(lhs), Body::Stage(rhs)) => {
            let mut lhs_bodys = lhs.bodys.clone();
            let mut subs_var = SubstituteVar::new();
            for (new_shape_idx, old_shape_idx) in lhs_replace.iter() {
                subs_var.add_replacement(
                    Variable::new(format!("ax{}", old_shape_idx)),
                    Variable::new(format!("ax{}", new_shape_idx))
                );
            }
            for i in lhs_bodys.iter_mut() {
                i.accept_mutate(&mut subs_var);
            }
            let lhs_new_axes = Arc::new(
                lhs_new_axes
                    .iter()
                    .map(|x| { Variable::new(format!("ax{}", x)).into() })
                    .collect::<Vec<PrimeExpr>>()
            );
            let mut insert_axes = InsertAxes::new(lhs_new_axes.clone(), output_id);
            for i in lhs_bodys.iter_mut() {
                insert_axes.set_expr(PrimeExpr::None);
                insert_axes.set_stmt(Stmt::None);
                i.insert_new_axes(&mut insert_axes);
            }

            let mut rhs_bodys = rhs.bodys.clone();
            let mut subs_var = SubstituteVar::new();
            for (new_shape_idx, old_shape_idx) in rhs_replace.iter() {
                subs_var.add_replacement(
                    Variable::new(format!("ax{}", old_shape_idx)),
                    Variable::new(format!("ax{}", new_shape_idx))
                );
            }
            for i in rhs_bodys.iter_mut() {
                i.accept_mutate(&mut subs_var);
            }
            let rhs_new_axes = Arc::new(
                rhs_new_axes
                    .iter()
                    .map(|x| { Variable::new(format!("ax{}", x)).into() })
                    .collect::<Vec<PrimeExpr>>()
            );
            let mut insert_axes = InsertAxes::new(rhs_new_axes.clone(), output_id);
            for i in rhs_bodys.iter_mut() {
                insert_axes.set_expr(PrimeExpr::None);
                insert_axes.set_stmt(Stmt::None);
                i.insert_new_axes(&mut insert_axes);
            }

            lhs_bodys.extend(rhs_bodys);
            lhs_bodys.push(post());
            let stage = Stage {
                dims: dims.clone(),
                bodys: lhs_bodys,
                id: output_id,
                out_id: output_id,
                dtype: ty_infer(lhs.dtype, rhs.dtype),
            };
            Body::Stage(stage)
        }
        _ => panic!("input is not a stage"),
    }
}

pub fn common_binop_out<F>(
    dims: &Vec<IterVar>,
    lhs_id: usize,
    rhs_id: usize,
    output_id: usize,
    binop: F
) -> Body
    where F: Fn(PrimeExpr, PrimeExpr) -> PrimeExpr
{
    Body::Stmt(
        StoreStmt::make(
            &Variable::make(&format!("%{}", output_id)),
            dims
                .iter()
                .enumerate()
                .map(
                    |(idx, x)|
                        x.var().to_prime_expr() *
                        Load::make(&format!("%{}.s", output_id), idx).into()
                )
                .reduce(|acc, x| acc + x)
                .unwrap(),
            binop(
                Variable::make(&format!("%{}_val", lhs_id)).into(),
                Variable::make(&format!("%{}_val", rhs_id)).into()
            )
        ).into()
    )
}

pub fn common_binop_in<F>(lhs_id: usize, rhs_id: usize, binop: F, output_id: usize) -> Body
    where F: Fn(PrimeExpr, PrimeExpr) -> PrimeExpr
{
    Body::Stmt(
        Stmt::LetStmt(
            LetStmt::make(
                &Variable::make(&format!("%{}_val", output_id)),
                binop(
                    Variable::make(&format!("%{}_val", lhs_id)).into(),
                    Variable::make(&format!("%{}_val", rhs_id)).into()
                ),
                false,
                Stmt::None
            )
        ).into()
    )
}

pub fn common_uaryop(
    is_output: bool,
    inputs: &Vec<Body>,
    shape: &Vec<PrimeExpr>,
    ty_infer: fn(Dtype) -> Dtype,
    method_name: &str,
    output_id: usize
) -> Body {
    if is_output {
        common_unaryop_helper(
            inputs,
            shape,
            output_id,
            ty_infer,
            |dims: &Vec<IterVar>, stage_out_id: usize|
                Body::Stmt(
                    Stmt::StoreStmt(
                        StoreStmt::make(
                            &Variable::make(&format!("%{}", output_id)),
                            dims
                                .iter()
                                .enumerate()
                                .map(
                                    |(idx, x)|
                                        x.var().to_prime_expr() *
                                        Load::make(&format!("%{}.s", output_id), idx).into()
                                )
                                .reduce(|acc, x| acc + x)
                                .unwrap(),
                            Call::make(
                                method_name,
                                &[Variable::make(&format!("%{}_val", stage_out_id))]
                            )
                        )
                    )
                )
        )
    } else {
        common_unaryop_helper(inputs, shape, output_id, ty_infer, |_, stage_out_id: usize|
            Body::Stmt(
                Stmt::LetStmt(
                    LetStmt::make(
                        &Variable::make(&format!("%{}_val", output_id)),
                        Call::make(
                            method_name,
                            &[Variable::make(&format!("%{}_val", stage_out_id))]
                        ),
                        false,
                        Stmt::None
                    )
                ).into()
            )
        )
    }
}

pub fn common_unaryop_helper<F>(
    inputs: &Vec<Body>,
    shape: &Vec<PrimeExpr>,
    output_id: usize,
    ty_infer: fn(Dtype) -> Dtype,
    post: F
) -> Body
    where F: Fn(&Vec<IterVar>, usize) -> Body
{
    match &inputs[0] {
        Body::Stage(stage) => {
            let mut stage = stage.clone();
            let dims = (0..shape.len())
                .map(|x| IterVar::new(0i64, shape[x].clone(), 1i64, &format!("ax{}", x)))
                .collect::<Vec<IterVar>>();
            stage.broadcast_new_dims(
                &(0..shape.len())
                    .map(|x| Load::make(Variable::make(&format!("%{}.s", output_id)), x).into())
                    .collect::<Vec<PrimeExpr>>(),
                &(0..shape.len())
                    .map(|x| Variable::make(&format!("ax{}", x)).into())
                    .collect::<Vec<PrimeExpr>>()
            );
            stage.bodys.push(post(&dims, stage.out_id));
            let stage = Stage {
                dims,
                bodys: stage.bodys.clone(),
                id: output_id,
                out_id: stage.out_id,
                dtype: ty_infer(stage.dtype),
            };
            Body::Stage(stage)
        }
        _ => panic!("input is not a stage"),
    }
}
