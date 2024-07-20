use std::{ collections::HashMap, sync::Arc };

use crate::{ halide::{ prime_expr::PrimeExpr, stmt::Stmt, variable::Variable }, iter_var::IterVar };

use super::operation::Operation;

pub struct SrgNode {
    pub(crate) id: usize,
    pub(crate) shape: Vec<PrimeExpr>,
    pub(crate) inputs: Arc<Vec<SrgNode>>,
    pub(crate) op: Operation,
    pub(crate) reduced_dim: usize,
    pub(crate) strides_cal: Arc<dyn Fn(&HashMap<String, i64>) -> Vec<Vec<i64>>>,
}
