use crate::halide::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, MutatorGetSet },
};

pub struct TransposeAxes {
    pub(crate) stmt: Stmt,
    pub(crate) expr: PrimeExpr,
    pub(crate) axes: Vec<usize>,
}

impl TransposeAxes {
    pub fn new(axes: Vec<usize>) -> Self {
        Self {
            stmt: Stmt::None,
            expr: PrimeExpr::None,
            axes,
        }
    }
}

impl MutatorGetSet for TransposeAxes {
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

impl IRMutateVisitor for TransposeAxes {
    fn visit_tensor_load(&mut self, tensor_load: &crate::halide::tensor_load::TensorLoad) {
        let mut new_begins = vec![];
        let mut new_steps = vec![];
        let mut new_axes = vec![];
        for i in 0..self.axes.len() {
            new_begins.push(tensor_load.begins[self.axes[i]].clone());
            new_steps.push(tensor_load.steps[self.axes[i]].clone());
            new_axes.push(tensor_load.axes[self.axes[i]].clone());
        }
        new_begins.extend_from_slice(&tensor_load.begins[self.axes.len()..]);
        new_steps.extend_from_slice(&tensor_load.steps[self.axes.len()..]);
        new_axes.extend_from_slice(&tensor_load.axes[self.axes.len()..]);
        self.set_expr(
            crate::halide::tensor_load::TensorLoad::make(
                tensor_load.var.as_ref(),
                new_begins,
                new_axes,
                new_steps,
                tensor_load.strides.as_ref().clone(),
                tensor_load.hints.as_ref().clone()
            )
        );
    }
}
