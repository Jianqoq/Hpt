use std::{ collections::HashMap, sync::Arc };

use crate::{ halide::{ prime_expr::PrimeExpr, stmt::Stmt, variable::Variable }, iter_var::IterVar };

pub struct SrgNode {
    pub(crate) id: usize,
    pub(crate) shape: Vec<PrimeExpr>,
    pub(crate) red_axes: Vec<IterVar>,
    pub(crate) inputs: Arc<Vec<SrgNode>>,
    pub(crate) stmts: Vec<Stmt>,
    pub(crate) strides_cal: Arc<dyn Fn(&HashMap<String, i64>) -> Vec<i64>>,
    pub(crate) using: Vec<usize>,
}
