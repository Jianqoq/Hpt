use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        alloca_stmt::AllocaStmt,
        exprs::{ BitAnd, Ge, Load, Lt },
        let_stmt::LetStmt,
        passes::const_fold::ConstFold,
        prime_expr::PrimeExpr,
        primitive_type::PrimitiveType,
        stmt::Stmt,
        traits::{ IRMutateVisitor, MutatorGetSet },
        utils::{ all, store_with_dims, store_with_idx },
        variable::Variable,
    },
    iter_var::IterVar,
    te::{ context::Context, stages::{ Body, If, Stage }, tensor::{ StridesCal, Tensor } },
    to_prim_expr::ToPrimeExpr,
};

pub fn pad_common(
    inputs: Vec<Body>,
    shape: Arc<Vec<PrimeExpr>>,
    pad_val: PrimeExpr,
    paddings: &[(PrimeExpr, PrimeExpr)],
    is_output: bool,
    id: usize
) -> Body {
    let dims = shape
        .iter()
        .enumerate()
        .map(|(idx, x)| {
            IterVar::new(0i64, x.clone(), 1i64, &Variable::new(format!("ax{}", idx)))
        })
        .collect::<Vec<IterVar>>();
    if is_output {
        if let Body::Stage(stage) = &inputs[0] {
            let store_dims = stage.dims
                .iter()
                .map(|x| x.var().to_prime_expr())
                .collect::<Vec<PrimeExpr>>();
            let store_strides = (0..stage.dims.len())
                .map(|x| { Load::make(&format!("%{}.s", id), x).into() })
                .collect::<Vec<PrimeExpr>>();
            pad_common_helper(
                is_output,
                dims,
                stage,
                paddings,
                id,
                vec![
                    Body::Stmt(
                        store_with_dims(
                            format!("%{}", id),
                            store_dims.clone(),
                            store_strides.clone(),
                            Variable::make(&format!("%{}_val", stage.out_id))
                        )
                    )
                ],
                vec![
                    Body::Stmt(
                        store_with_dims(format!("%{}", id), store_dims, store_strides, pad_val)
                    )
                ]
            )
        } else {
            panic!("pad_common: input is not a stage");
        }
    } else {
        if let Body::Stage(stage) = &inputs[0] {
            pad_common_helper(
                is_output,
                dims,
                stage,
                paddings,
                id,
                vec![
                    Body::Stmt(
                        store_with_idx(
                            format!("%{}_val_ptr", id),
                            0i64,
                            Variable::make(&format!("%{}_val", stage.out_id))
                        )
                    )
                ],
                vec![Body::Stmt(store_with_idx(format!("%{}_val_ptr", id), 0i64, pad_val))]
            )
        } else {
            panic!("pad_common: input is not a stage");
        }
    }
}

pub fn pad_common_helper(
    is_output: bool,
    dims: Vec<IterVar>,
    input: &Stage,
    padding: &[(PrimeExpr, PrimeExpr)],
    out_id: usize,
    true_bodys: Vec<Body>,
    false_bodys: Vec<Body>
) -> Body {
    let mut conds = vec![];
    for (i, (begin, end)) in padding.iter().enumerate() {
        let gt: PrimeExpr = Ge::make(Variable::new(format!("ax{}", i)), begin).into();
        let lt: PrimeExpr = Lt::make(Variable::new(format!("ax{}", i)), dims[i].end() - end).into();
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

    let mut bodys = if is_output {
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

    let mut pad_visitor = PadVisitor::new(
        padding
            .iter()
            .map(|(x, _)| x.clone())
            .collect()
    );
    for body in bodys.iter_mut() {
        body.accept_mutate(&mut pad_visitor);
    }
    let stage = Stage {
        dims,
        bodys,
        id: out_id,
        out_id,
        dtype: input.dtype,
        begins: input.begins.clone(),
        steps: input.steps.clone(),
        axes: input.axes.clone(),
    };
    Body::Stage(stage)
}

pub struct PadVisitor {
    pub(crate) stmt: Stmt,
    pub(crate) expr: PrimeExpr,
    pub(crate) offsets: Vec<PrimeExpr>,
}

impl PadVisitor {
    pub fn new(offsets: Vec<PrimeExpr>) -> Self {
        Self { stmt: Stmt::None, expr: PrimeExpr::None, offsets }
    }
}

impl MutatorGetSet for PadVisitor {
    fn set_expr<T: Into<PrimeExpr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into();
    }

    fn expr(&self) -> &PrimeExpr {
        &self.expr
    }

    fn stmt(&self) -> &Stmt {
        &self.stmt
    }
}

impl IRMutateVisitor for PadVisitor {
    fn visit_tensor_load(&mut self, tensor_load: &crate::halide::tensor_load::TensorLoad) {
        let mut new_begins = vec![];
        let kept_begins = tensor_load.begins[self.offsets.len()..].to_vec();
        for (offset, old_begin) in self.offsets.iter().zip(tensor_load.begins.iter()) {
            new_begins.push(old_begin.clone() - offset.clone());
        }
        new_begins.extend(kept_begins);
        self.set_expr(crate::halide::tensor_load::TensorLoad {
            var: tensor_load.var.clone(),
            begins: new_begins.into(),
            axes: tensor_load.axes.clone(),
            steps: tensor_load.steps.clone(),
            strides: tensor_load.strides.clone(),
            hints: tensor_load.hints.clone(),
        });
    }
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
        assert_eq!(padding.len(), a.shape.len());
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
            shape: shape.clone(),
            dtype: a.dtype,
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                let prev_fn = prev_fn[0].clone();
                Arc::new(move |map: &HashMap<Arc<String>, i64>| { prev_fn(map) })
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                pad_common(inputs, shape.clone(), pad_val.as_ref().clone(), &padding, is_output, id)
            }),
            id,
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
