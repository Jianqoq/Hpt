use super::{ exprs::BitAnd, prime_expr::PrimeExpr };

pub fn all(conds: &[PrimeExpr]) -> PrimeExpr {
    conds
        .into_iter()
        .cloned()
        .reduce(|acc, x| BitAnd::make(acc, x).into())
        .unwrap()
}
