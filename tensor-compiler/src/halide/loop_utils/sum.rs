use crate::halide::{ exprs::Reduce, prime_expr::PrimeExpr, variable::Variable };

pub fn sum<
    A: IntoIterator<Item: Into<PrimeExpr>>,
    B: IntoIterator<Item: Into<PrimeExpr>>,
    C: IntoIterator<Item: Into<PrimeExpr>>,
    D: IntoIterator<Item: Into<PrimeExpr>>,
    E: IntoIterator<Item: Into<PrimeExpr>>
>(
    expr: A,
    init: B,
    start: C,
    end: D,
    step: E,
    names: Vec<&str>
) -> PrimeExpr {
    Reduce::make(
        expr.into_iter().map(|x| x.into()).collect(),
        init.into_iter().map(|x| x.into()).collect(),
        start.into_iter().map(|x| x.into()).collect(),
        end.into_iter().map(|x| x.into()).collect(),
        step.into_iter().map(|x| x.into()).collect(),
        names
            .iter()
            .map(|x| Variable::new(x.to_string()).into())
            .collect(),
        "sum"
    ).into()
}
