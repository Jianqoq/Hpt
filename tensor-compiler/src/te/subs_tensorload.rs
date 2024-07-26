use crate::halide::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    tensor_load::TensorLoad,
    traits::{ IRMutateVisitor, MutatorGetSet },
};

pub struct SubsTensorLoadDims<'a> {
    pub(crate) expr: PrimeExpr,
    pub(crate) stmt: Stmt,
    pub(crate) strides: &'a Vec<PrimeExpr>,
    pub(crate) axes: &'a Vec<PrimeExpr>,
}

impl<'a> SubsTensorLoadDims<'a> {
    pub fn new(strides: &'a Vec<PrimeExpr>, axes: &'a Vec<PrimeExpr>) -> Self {
        Self {
            expr: PrimeExpr::None,
            stmt: Stmt::None,
            strides,
            axes,
        }
    }
}

impl<'a> MutatorGetSet for SubsTensorLoadDims<'a> {
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

impl<'a> IRMutateVisitor for SubsTensorLoadDims<'a> {
    fn visit_tensor_load(&mut self, tensor_load: &TensorLoad) {
        let begins = &tensor_load.begins;
        let steps = &tensor_load.steps;
        let strides = &tensor_load.strides;
        let axes = &tensor_load.axes;
        if
            self.strides.len() != strides.len() ||
            self.axes.len() != axes.len()
        {
            self.set_expr(
                TensorLoad::make(
                    tensor_load.var.as_ref(),
                    tensor_load.begins.as_ref().clone(),
                    self.axes.clone(),
                    tensor_load.steps.as_ref().clone(),
                    self.strides.clone(),
                    tensor_load.hints.as_ref().clone()
                )
            );
        } else if
            self.strides
                .iter()
                .zip(strides.iter())
                .all(|(x, y)| x == y) &&
            self.axes
                .iter()
                .zip(axes.iter())
                .all(|(x, y)| x == y)
        {
            self.set_expr(tensor_load.clone());
        } else {
            self.set_expr(
                TensorLoad::make(
                    tensor_load.var.as_ref(),
                    tensor_load.begins.as_ref().clone(),
                    self.axes.clone(),
                    tensor_load.steps.as_ref().clone(),
                    self.strides.clone(),
                    tensor_load.hints.as_ref().clone()
                )
            );
        }
    }
}
