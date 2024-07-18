use crate::{ halide::{ exprs::Reduce, prime_expr::PrimeExpr }, iter_var::IterVar };
use crate::to_prim_expr::ToPrimeExpr;
macro_rules! register_reduce {
    ($op:ident) => {
        pub fn $op<
    A: IntoIterator<Item: Into<PrimeExpr>>,
    C: IntoIterator<Item: Into<IterVar>>
>(expr: A, init: &[&dyn ToPrimeExpr], iter_vars: C) -> PrimeExpr {
            Reduce::make(
                expr
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
                init
                    .into_iter()
                    .map(|x| x.to_prime_expr())
                    .collect(),
                iter_vars
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
                stringify!($op)
            ).into()
        }
    };
}

register_reduce!(sum);
register_reduce!(prod);
register_reduce!(max);
register_reduce!(min);
register_reduce!(argmax);
register_reduce!(argmin);
