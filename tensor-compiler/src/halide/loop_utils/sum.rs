use crate::halide::{ exprs::Reduce, prime_expr::PrimeExpr, variable::Variable };

pub fn sum(
    expr: PrimeExpr,
    init: PrimeExpr,
    start: PrimeExpr,
    end: PrimeExpr,
    step: PrimeExpr,
    name: &str
) -> PrimeExpr {
    Reduce::make(expr, init, start, end, step, Variable::make(name).into()).into()
}
