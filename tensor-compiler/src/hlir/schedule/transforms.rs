use std::sync::Arc;

use crate::{halide::prime_expr::PrimeExpr, iter_var::IterVar};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transforms {
    Split(Arc<String>, usize, PrimeExpr),
    Fuse(Arc<String>, usize, usize),
    Reorder(Vec<PrimeExpr>),
    Inline(Arc<String>),
    ComputeAt(Arc<String>, PrimeExpr),
    Tile(Vec<PrimeExpr>),
}
