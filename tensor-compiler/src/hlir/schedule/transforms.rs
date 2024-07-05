use std::sync::Arc;

use crate::{halide::prime_expr::PrimeExpr, iter_val::_IterVar};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transforms {
    Split(_IterVar, PrimeExpr),
    Fuse(Vec<PrimeExpr>),
    Reorder(Vec<PrimeExpr>),
    Inline(Arc<String>),
    ComputeAt(Arc<String>, PrimeExpr),
    Tile(Vec<PrimeExpr>),
}
