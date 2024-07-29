use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        assign_stmt::AssignStmt,
        exprs::{ BitAnd, Ge, Load, Lt },
        passes::const_fold::ConstFold,
        prime_expr::PrimeExpr,
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
            pad_common_helper(
                stage,
                paddings,
                id,
                Body::Stmt(
                    StoreStmt::make(
                        &Variable::make(&format!("%{}", id)),
                        stage.dims
                            .iter()
                            .enumerate()
                            .map(
                                |(idx, x)|
                                    x.var().to_prime_expr() *
                                    Load::make(&format!("%{}.s", id), idx).into()
                            )
                            .reduce(|acc, x| acc + x)
                            .unwrap(),
                        Variable::make(&format!("%{}_val", stage.out_id))
                    ).into()
                ),
                Body::Stmt(
                    StoreStmt::make(
                        &Variable::make(&format!("%{}", id)),
                        stage.dims
                            .iter()
                            .enumerate()
                            .map(
                                |(idx, x)|
                                    x.var().to_prime_expr() *
                                    Load::make(&format!("%{}.s", id), idx).into()
                            )
                            .reduce(|acc, x| acc + x)
                            .unwrap(),
                        pad_val.clone()
                    ).into()
                )
            )
        } else {
            panic!("pad_common: input is not a stage");
        }
    } else {
        if let Body::Stage(stage) = &inputs[0] {
            pad_common_helper(
                stage,
                paddings,
                id,
                Body::Stmt(
                    Stmt::AssignStmt(
                        AssignStmt::make(
                            &Variable::make(&format!("%{}_val", id)),
                            Variable::make(&format!("%{}_val", stage.out_id))
                        )
                    ).into()
                ),
                Body::Stmt(
                    Stmt::AssignStmt(
                        AssignStmt::make(&Variable::make(&format!("%{}_val", id)), pad_val.clone())
                    ).into()
                )
            )
        } else {
            panic!("pad_common: input is not a stage");
        }
    }
}

pub fn pad_common_helper(
    input: &Stage,
    padding: &[(PrimeExpr, PrimeExpr)],
    out_id: usize,
    true_body: Body,
    false_body: Body
) -> Body {
    let mut conds = vec![];
    for (i, (begin, end)) in padding.iter().enumerate() {
        let gt: PrimeExpr = Ge::make(Variable::new(format!("ax{}", i)), begin).into();
        let lt: PrimeExpr = Lt::make(Variable::new(format!("ax{}", i)), end).into();
        let and: PrimeExpr = BitAnd::make(gt, lt).into();
        conds.push(and);
    }
    let cond = all(&conds);
    let mut true_bodys = input.bodys.clone();
    true_bodys.push(true_body);

    let if_then_else = Body::If(If {
        cond,
        true_bodys,
        false_bodys: vec![false_body],
        id: out_id,
        input: input.id,
    });

    let stage = Stage {
        dims: input.dims.clone(),
        bodys: vec![if_then_else],
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
