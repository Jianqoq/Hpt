use std::sync::Arc;

use crate::halide::prime_expr::PrimeExpr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transforms {
    Split(PrimeExpr, PrimeExpr),
    Fuse(Vec<PrimeExpr>),
    Reorder(Vec<PrimeExpr>),
    ComputeInline(Arc<String>),
    ComputeAt(Arc<String>, PrimeExpr),
    Tile(Vec<PrimeExpr>),
}
