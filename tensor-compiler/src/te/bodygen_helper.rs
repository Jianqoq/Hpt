use crate::{
    halide::{
        alloca_stmt::AllocaStmt,
        exprs::{ Add, Load },
        inplace_store_stmt::InplaceAdd,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        primitive_type::PrimitiveType,
        stmt::Stmt,
        store_stmt::StoreStmt,
        variable::Variable,
    },
    iter_var::IterVar,
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
            common_reduce_helper(original_shape, axes, init, output_id, stage, |_| vec![
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
            ])
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
