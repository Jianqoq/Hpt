use crate::halide::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, MutatorGetSet },
};

pub struct SliceVisitor {
    pub(crate) stmt: Stmt,
    pub(crate) expr: PrimeExpr,
    pub(crate) begins: Vec<PrimeExpr>,
    pub(crate) steps: Vec<PrimeExpr>,
}

impl SliceVisitor {
    pub fn new(begins: Vec<PrimeExpr>, steps: Vec<PrimeExpr>) -> Self {
        Self {
            stmt: Stmt::None,
            expr: PrimeExpr::None,
            begins,
            steps,
        }
    }
}

impl MutatorGetSet for SliceVisitor {
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

impl IRMutateVisitor for SliceVisitor {
    fn visit_tensor_load(&mut self, tensor_load: &crate::halide::tensor_load::TensorLoad) {
        let mut new_begins = vec![];
        let mut new_steps = vec![];
        for i in 0..self.begins.len() {
            new_begins.push(self.begins[i].clone() + tensor_load.begins[i].clone());
            new_steps.push(self.steps[i].clone() * tensor_load.steps[i].clone());
        }
        new_begins.extend_from_slice(&tensor_load.begins[self.begins.len()..]);
        new_steps.extend_from_slice(&tensor_load.steps[self.begins.len()..]);
        self.set_expr(
            crate::halide::tensor_load::TensorLoad::make(
                tensor_load.var.as_ref(),
                new_begins,
                tensor_load.axes.as_ref().clone(),
                new_steps,
                tensor_load.strides.as_ref().clone(),
                tensor_load.hints.as_ref().clone()
            )
        );
    }
}
