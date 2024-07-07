use std::sync::Arc;

use crate::halide::prime_expr::PrimeExpr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transforms {
    Split(Arc<String>, usize, PrimeExpr),
    Fuse(Arc<String>, usize, usize),
    Reorder(Arc<String>, Vec<usize>),
    Inline(Arc<String>),
    ComputeAt(Arc<String>, PrimeExpr),
    Tile(Vec<PrimeExpr>),
}
