use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::halide::prime_expr::PrimeExpr;

use super::{ hstrides::HStrides, operation::Operation };

#[derive(Clone)]
pub struct SrgNode {
    pub(crate) id: usize,
    pub(crate) shape: Arc<Vec<PrimeExpr>>,
    pub(crate) inputs: Arc<Vec<usize>>,
    pub(crate) op: Operation,
    pub(crate) strides_cal: Arc<dyn Fn(&HashMap<String, i64>) -> Vec<HStrides>>,
    pub(crate) span: &'static Location<'static>,
}
