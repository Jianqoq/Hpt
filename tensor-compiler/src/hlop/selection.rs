use crate::{ tensor::Tensor, to_prim_expr::ToPrimeExpr };

impl Tensor {
    pub fn select(&self, _: &[(&dyn ToPrimeExpr, &dyn ToPrimeExpr, &dyn ToPrimeExpr)]) {

    }
}
