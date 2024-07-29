use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        alloca_stmt::AllocaStmt,
        assign_stmt::AssignStmt,
        exprs::{ BitAnd, Ge, Load, Lt },
        let_stmt::LetStmt,
        passes::const_fold::ConstFold,
        prime_expr::PrimeExpr,
        primitive_type::PrimitiveType,
        stmt::Stmt,
        store_stmt::StoreStmt,
        utils::all,
        variable::Variable,
    },
    te::{ context::Context, stages::{ Body, If, Stage }, tensor::{ StridesCal, Tensor } },
    to_prim_expr::ToPrimeExpr,
};

pub fn pad_common(
    inputs: Vec<Body>,
    pad_val: PrimeExpr,
    paddings: &[(PrimeExpr, PrimeExpr)],
    is_output: bool,
    id: usize
) -> Body {
    if is_output {
        if let Body::Stage(stage) = &inputs[0] {
            let store_indices = stage.dims
                .iter()
                .enumerate()
                .map(|(idx, x)| {
                    x.var().to_prime_expr() * Load::make(&format!("%{}.s", id), idx).into()
                })
                .reduce(|acc, x| acc + x)
                .unwrap_or((0i64).into());
            let store_var = Variable::new(format!("%{}", id));

            pad_common_helper(
                is_output,
                stage,
                paddings,
                id,
                vec![
                    Body::Stmt(
                        StoreStmt::make(
                            &store_var,
                            &store_indices,
                            Variable::make(&format!("%{}_val", stage.out_id))
                        ).into()
                    )
                ],
                vec![
                    Body::Stmt(StoreStmt::make(&store_var, &store_indices, pad_val.clone()).into())
                ]
            )
        } else {
            panic!("pad_common: input is not a stage");
        }
    } else {
        if let Body::Stage(stage) = &inputs[0] {
            pad_common_helper(
                is_output,
                stage,
                paddings,
                id,
                vec![
                    Body::Stmt(
                        Stmt::StoreStmt(
                            StoreStmt::make(
                                &Variable::make(&format!("%{}_val_ptr", id)),
                                0i64,
                                Variable::make(&format!("%{}_val", stage.out_id))
                            )
                        ).into()
                    )
                ],
                vec![
                    Body::Stmt(
                        Stmt::StoreStmt(
                            StoreStmt::make(
                                &Variable::make(&format!("%{}_val_ptr", id)),
                                0i64,
                                pad_val.clone()
                            )
                        ).into()
                    )
                ]
            )
        } else {
            panic!("pad_common: input is not a stage");
        }
    }
}

pub fn pad_common_helper(
    is_output: bool,
    input: &Stage,
    padding: &[(PrimeExpr, PrimeExpr)],
    out_id: usize,
    true_bodys: Vec<Body>,
    false_bodys: Vec<Body>
) -> Body {
    let mut conds = vec![];
    for (i, (begin, end)) in padding.iter().enumerate() {
        let gt: PrimeExpr = Ge::make(Variable::new(format!("ax{}", i)), begin).into();
        let lt: PrimeExpr = Lt::make(Variable::new(format!("ax{}", i)), end).into();
        let and: PrimeExpr = BitAnd::make(gt, lt).into();
        conds.push(and);
    }
    let cond = all(&conds);

    let mut _true_bodys = input.bodys.clone();
    _true_bodys.extend(true_bodys);

    let if_then_else = Body::If(If {
        cond,
        true_bodys: _true_bodys,
        false_bodys,
        id: out_id,
        input: input.id,
    });

    let bodys = if is_output {
        vec![if_then_else]
    } else {
        let init = Body::Stmt(
            AllocaStmt::make(
                &Variable::new(format!("%{}_val_ptr", out_id)),
                PrimitiveType::Dtype(input.dtype),
                1,
                Stmt::None
            ).into()
        );
        let let_ = Body::Stmt(
            LetStmt::make(
                &Variable::new(format!("%{}_val", out_id)),
                Load::make(&Variable::new(format!("%{}_val_ptr", out_id)), 0),
                false,
                Stmt::None
            ).into()
        );
        vec![init, if_then_else, let_]
    };

    let stage = Stage {
        dims: input.dims.clone(),
        bodys,
        id: out_id,
        out_id,
        dtype: input.dtype,
    };
    Body::Stage(stage)
}

impl Context {
    pub fn pad(
        &mut self,
        a: &Tensor,
        padding: &[(&dyn ToPrimeExpr, &dyn ToPrimeExpr)],
        pad_value: &dyn ToPrimeExpr
    ) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        assert!(padding.len() == a.shape.len());
        let mut new_shape = a.shape
            .iter()
            .zip(padding.iter())
            .map(|(x, (begin, end))| x + begin.to_prime_expr() + end.to_prime_expr())
            .collect::<Vec<PrimeExpr>>();
        new_shape.iter_mut().for_each(|x| {
            let mut const_fold = ConstFold::new();
            *x = const_fold.const_fold(x.clone());
        });
        let new_shape = Arc::new(new_shape);

        let shape = new_shape.clone();

        let padding = padding
            .iter()
            .map(|(begin, end)| { (begin.to_prime_expr().clone(), end.to_prime_expr().clone()) })
            .collect::<Vec<(PrimeExpr, PrimeExpr)>>();

        let pad_val = Arc::new(pad_value.to_prime_expr());
        let ret = Tensor {
            shape,
            dtype: a.dtype,
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                let prev_fn = prev_fn[0].clone();
                Arc::new(move |map: &HashMap<Arc<String>, i64>| { prev_fn(map) })
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                pad_common(inputs, pad_val.as_ref().clone(), &padding, is_output, id)
            }),
            id,
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
