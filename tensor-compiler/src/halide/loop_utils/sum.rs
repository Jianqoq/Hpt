use crate::halide::{ exprs::Reduce, prime_expr::PrimeExpr, variable::Variable };

pub fn sum(
    expr: PrimeExpr,
    init: PrimeExpr,
    start: Vec<PrimeExpr>,
    end: Vec<PrimeExpr>,
    step: Vec<PrimeExpr>,
    names: Vec<&str>
) -> PrimeExpr {
    Reduce::make(
        expr,
        init,
        start,
        end,
        step,
        names
            .iter()
            .map(|x| Variable::new(x.to_string()).into())
            .collect()
    ).into()
}
