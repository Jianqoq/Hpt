use std::{ collections::HashMap, panic::Location, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;
use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        tensor_load::TensorLoad,
        utils::var,
        variable::Variable,
    },
    iter_var::IterVar,
    te::idx_evaluator::IdxEvaluator,
};

use super::{ hstrides::HStrides, stages::{ Body, Stage } };

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

impl Tensor {
    pub fn id(&self) -> usize {
        self.id
    }
    pub fn as_input(&mut self) {
        let shape = self.shape.clone();
        let shape1 = shape.clone();
        let dtype = self.dtype;
        self.strides_cal = Arc::new(move |_: Vec<StridesCal>| {
            let shape = shape.clone();
            Arc::new(move |map: &HashMap<Arc<String>, i64>| {
                let real_shape = shape
                    .iter()
                    .map(|x| { IdxEvaluator::new(map).eval(x) })
                    .collect::<Vec<i64>>();
                let hstrides = HStrides {
                    strides: shape_to_strides(&real_shape).inner().clone(),
                    reduced_dim: 0,
                    offset: 0,
                };
                vec![hstrides]
            })
        });
        self.body_gen = Arc::new(move |_: Vec<Body>, _: bool, id: usize| {
            let shape = shape1.clone();
            let body = Body::Stmt(
                Stmt::LetStmt(
                    LetStmt::make(
                        &Variable::make(&format!("%{}_val", id)),
                        TensorLoad {
                            var: Variable::make(&format!("%{}", id)).into(),
                            begins: (0..shape.len())
                                .map(|_| (0i64).into())
                                .collect::<Vec<PrimeExpr>>()
                                .into(),
                            axes: (0..shape.len())
                                .map(|x| Variable::make(&format!("ax{}", x)).into())
                                .collect::<Vec<PrimeExpr>>()
                                .into(),
                            steps: (0..shape.len())
                                .map(|_| (1i64).into())
                                .collect::<Vec<PrimeExpr>>()
                                .into(),
                            strides: (0..shape.len())
                                .map(|idx|
                                    Load::make(Variable::make(&format!("%{}.s", id)), idx).into()
                                )
                                .collect::<Vec<PrimeExpr>>()
                                .into(),
                            hints: vec![].into(),
                            flag: Default::default(),
                        },
                        false,
                        Stmt::None
                    )
                ).into()
            );
            let stage = Stage {
                dims: (0..shape.len())
                    .map(|x| IterVar::new(0i64, shape[x].clone(), 1i64, &format!("ax{}", x)))
                    .collect(),
                bodys: vec![body],
                id,
                out_id: id,
                dtype,
                steps: vec![1i64.into(); shape.len()],
                begins: vec![0i64.into(); shape.len()],
                axes: (0..shape.len()).map(|x| var(format!("ax{}", x)).into()).collect(),
            };
            Body::Stage(stage)
        });
        self.inputs = Arc::new(vec![]);
    }
}
