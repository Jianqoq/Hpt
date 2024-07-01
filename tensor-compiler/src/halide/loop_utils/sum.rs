use crate::{ halide::{ exprs::Reduce, prime_expr::PrimeExpr }, iter_val::IterVar };

pub fn sum<
    A: IntoIterator<Item: Into<PrimeExpr>>,
    B: IntoIterator<Item: Into<PrimeExpr>>,
    C: IntoIterator<Item: Into<IterVar>>
>(expr: A, init: B, iter_vars: C) -> PrimeExpr {
    Reduce::make(
        expr
            .into_iter()
            .map(|x| x.into())
            .collect(),
        init
            .into_iter()
            .map(|x| x.into())
            .collect(),
        iter_vars
            .into_iter()
            .map(|x| x.into())
            .collect(),
        "sum"
    ).into()
}
