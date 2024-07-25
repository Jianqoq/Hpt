use crate::halide::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    tensor_load::TensorLoad,
    traits::{ IRMutateVisitor, MutatorGetSet },
};

pub struct StridesVisitor {
    pub(crate) cnt: i64,
    pub(crate) expr: PrimeExpr,
    pub(crate) stmt: Stmt,
}

impl StridesVisitor {
    pub fn new() -> Self {
        Self {
            cnt: 0,
            expr: PrimeExpr::None,
            stmt: Stmt::None,
        }
    }
}

impl MutatorGetSet for StridesVisitor {
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

impl IRMutateVisitor for StridesVisitor {
    fn visit_tensor_load(&mut self, tensor_load: &crate::halide::tensor_load::TensorLoad) {
        self.set_expr(TensorLoad {
            var: tensor_load.var.clone(),
            axes: tensor_load.axes.clone(),
            strides: tensor_load.strides
                .iter()
                .map(|x| {
                    let mut load = x.to_load().unwrap().clone();
                    load.name = format!("strides{}", self.cnt).into();
                    load.into()
                })
                .collect::<Vec<PrimeExpr>>()
                .into(),
            hints: tensor_load.hints.clone(),
        });
        self.cnt += 1;
    }
}
