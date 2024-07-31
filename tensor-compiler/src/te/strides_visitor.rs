use crate::halide::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    store_stmt::StoreStmt,
    tensor_load::TensorLoad,
    traits::{ IRMutateVisitor, MutatorGetSet },
    variable::Variable,
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
    fn visit_tensor_load(&mut self, tensor_load: &TensorLoad) {
        self.set_expr(TensorLoad {
            var: tensor_load.var.clone(),
            begins: tensor_load.begins.clone(),
            steps: tensor_load.steps.clone(),
            axes: tensor_load.axes.clone(),
            strides: tensor_load.strides
                .iter()
                .map(|x| {
                    let mut load = x.to_load().unwrap().clone();
                    load.name = format!("istrides{}", self.cnt).into();
                    load.into()
                })
                .collect::<Vec<PrimeExpr>>()
                .into(),
            hints: tensor_load.hints.clone(),
        });
        self.cnt += 1;
    }
}

pub struct StridesStoreVisitor {
    pub(crate) cnt: i64,
    pub(crate) expr: PrimeExpr,
    pub(crate) stmt: Stmt,
    pub(crate) in_store: bool,
}

impl StridesStoreVisitor {
    pub fn new() -> Self {
        Self {
            cnt: 0,
            expr: PrimeExpr::None,
            stmt: Stmt::None,
            in_store: false,
        }
    }
}

impl MutatorGetSet for StridesStoreVisitor {
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

impl IRMutateVisitor for StridesStoreVisitor {
    fn visit_variable(&mut self, var: &Variable) {
        if self.in_store {
            if var.name.contains(".s") {
                self.set_expr(Variable::new(format!("ostrides{}", self.cnt)));
            } else {
                self.set_expr(var.clone());
            }
        } else {
            self.set_expr(var.clone());
        }
    }

    fn visit_store(&mut self, store: &StoreStmt) {
        self.in_store = true;
        let indices = self.mutate_expr(store.indices());
        self.set_stmt(StoreStmt {
            var: store.var.clone(),
            indices: indices.clone().into(),
            val: store.val.clone(),
        });
        self.in_store = false;
    }
}
