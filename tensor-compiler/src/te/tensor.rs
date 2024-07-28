use std::{ collections::HashMap, panic::Location, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::halide::prime_expr::PrimeExpr;

use super::{hstrides::HStrides, stages::Body};

pub type StridesCal = Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>;
pub type StridesCalHelper = Arc<dyn Fn(Vec<StridesCal>) -> StridesCal>;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) shape: Arc<Vec<PrimeExpr>>,
    pub(crate) dtype: Dtype,
    pub(crate) inputs: Arc<Vec<usize>>,
    pub(crate) span: &'static Location<'static>,
    pub(crate) strides_cal: StridesCalHelper,
    pub(crate) body_gen: Arc<dyn Fn(Vec<Body>, bool, usize) -> Body>,
    pub(crate) id: usize,
}
