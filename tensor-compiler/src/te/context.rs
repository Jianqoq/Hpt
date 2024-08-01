use std::{ collections::{ HashMap, HashSet }, panic::Location, sync::Arc };

use tensor_common::{
    shape::Shape,
    shape_utils::is_reshape_possible,
    strides_utils::shape_to_strides,
};
use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::{ Call, Int, Load },
        let_stmt::LetStmt,
        passes::const_fold::ConstFold,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        tensor_load::TensorLoad,
        traits::MutatorGetSet,
        utils::store_with_dims,
        variable::Variable,
    },
    iter_var::IterVar,
    te::{ hstrides::HStrides, idx_evaluator::IdxEvaluator },
    to_prim_expr::ToPrimeExpr,
};

use super::{
    rc_mut::RcMut,
    schedule::Schedule,
    slice_helper::SliceVisitor,
    srg::Srg,
    srg_node::SrgNode,
    stages::{ Body, Stage },
    strides_cal_helper::slice_strides_cal,
    tensor::{ StridesCal, Tensor },
};

#[derive(Clone)]
pub struct Context {
    pub(crate) nodes: RcMut<HashMap<usize, Tensor>>,
    pub(crate) vars: RcMut<HashSet<Variable>>,
    pub(crate) id: RcMut<usize>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: RcMut::new(HashMap::new()),
            id: RcMut::new(0),
            vars: RcMut::new(HashSet::new()),
        }
    }

    pub fn to_schedule(self, order: &[usize]) -> Schedule {
        let mut nodes = HashMap::new();
        for (id, node) in self.nodes.borrow().iter() {
            let srg_node = SrgNode {
                id: *id,
                shape: node.shape.clone(),
                inputs: node.inputs.clone(),
                outputs: Arc::new(
                    self.nodes
                        .borrow()
                        .iter()
                        .filter_map(|(k, v)| {
                            if v.inputs.contains(id) { Some(*k) } else { None }
                        })
                        .collect()
                ),
                strides_cal: Arc::new(|_| vec![]),
                span: node.span,
                dtype: node.dtype,
            };
            nodes.insert(*id, srg_node);
        }
        let mut srg = Srg {
            nodes,
            tensors: self.nodes.clone(),
        };
        srg.create_strides_cal(&order);
        srg.create_schedule(order)
    }

    #[track_caller]
    pub fn var(&mut self, name: &str) -> Variable {
        let var = Variable::new(name.to_string());
        self.vars.borrow_mut().insert(var.clone());
        var
    }

    #[track_caller]
    pub fn placeholder(&mut self, shape: &[&dyn ToPrimeExpr], dtype: Dtype) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let shape = Arc::new(
            shape
                .into_iter()
                .map(|x| x.to_prime_expr())
                .collect::<Vec<PrimeExpr>>()
        );
        let shape1 = shape.clone();
        let tensor = Tensor {
            shape: shape.clone(),
            dtype,
            span: Location::caller(),
            inputs: Arc::new(vec![]),
            id,
            strides_cal: Arc::new(move |_: Vec<StridesCal>| {
                let shape = shape1.clone();
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
            }),
            body_gen: Arc::new(move |_: Vec<Body>, _: bool, id: usize| {
                let shape = shape.clone();
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
                                        Load::make(
                                            Variable::make(&format!("%{}.s", id)),
                                            idx
                                        ).into()
                                    )
                                    .collect::<Vec<PrimeExpr>>()
                                    .into(),
                                hints: vec![].into(),
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
                    axes: (0..shape.len()).map(|x| format!("ax{}", x).into()).collect(),
                };
                Body::Stage(stage)
            }),
        };
        self.nodes.borrow_mut().insert(id, tensor.clone());
        tensor
    }

    #[track_caller]
    pub fn reshape(&mut self, a: &Tensor, shape: &[&dyn ToPrimeExpr]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let new_shape = Arc::new(
            shape
                .into_iter()
                .map(|x| x.to_prime_expr())
                .collect::<Vec<PrimeExpr>>()
        );
        let prev_shape = a.shape.clone();
        let shape1 = new_shape.clone();
        let tensor = Tensor {
            shape: new_shape.clone().into(),
            dtype: a.dtype.clone(),
            span: Location::caller(),
            inputs: Arc::new(vec![a.id]),
            id,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                let prev_fn = prev_fn[0].clone();
                let new_shape = shape1.clone();
                let prev_shape = prev_shape.clone();
                Arc::new(move |map: &HashMap<Arc<String>, i64>| {
                    let prev_strides = prev_fn.clone()(map);
                    let new_shape = new_shape
                        .iter()
                        .map(|x| { IdxEvaluator::new(map).eval(x) })
                        .collect::<Vec<i64>>();
                    let prev_shape = prev_shape
                        .iter()
                        .map(|x| { IdxEvaluator::new(map).eval(x) })
                        .collect::<Vec<i64>>();
                    let new_shape = Shape::from(new_shape);
                    let prev_shape = Shape::from(prev_shape);
                    let mut ret = vec![];
                    for i in prev_strides.into_iter() {
                        let masked_strides = &i[..i.strides.len() - i.reduced_dim];
                        assert_eq!(masked_strides.len(), prev_shape.len());
                        if
                            let Some(new_strides) = is_reshape_possible(
                                &prev_shape,
                                masked_strides,
                                &new_shape
                            )
                        {
                            let mut new = vec![];
                            for i in new_strides.inner().iter() {
                                new.push(*i);
                            }
                            for i in i[i.strides.len() - i.reduced_dim..].iter() {
                                new.push(*i);
                            }
                            assert_eq!(new.len(), new_shape.len() + i.reduced_dim);
                            let new = HStrides {
                                strides: new,
                                reduced_dim: i.reduced_dim,
                                offset: i.offset,
                            };
                            ret.push(new);
                        } else {
                            panic!("Reshape not possible, {}", Location::caller());
                        }
                    }
                    ret
                })
            }),
            /* reshape we will stop fusion as long as the tensorload is not pure ax * id.s[idx] */
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let shape = new_shape.clone();
                if is_output {
                    if let Body::Stage(stage) = &inputs[0] {
                        let mut stage = stage.clone();
                        let dims = (0..shape.len())
                            .map(|x|
                                IterVar::new(0i64, shape[x].clone(), 1i64, &format!("ax{}", x))
                            )
                            .collect::<Vec<IterVar>>();
                        stage.broadcast_new_dims(
                            &(0..shape.len())
                                .map(|x|
                                    Load::make(Variable::make(&format!("%{}.s", id)), x).into()
                                )
                                .collect::<Vec<PrimeExpr>>(),
                            &(0..shape.len())
                                .map(|x| Variable::make(&format!("ax{}", x)).into())
                                .collect::<Vec<PrimeExpr>>()
                        );
                        let body = Body::Stmt(
                            store_with_dims(
                                format!("%{}", id),
                                dims
                                    .iter()
                                    .map(|x| x.var().to_prime_expr())
                                    .collect::<Vec<PrimeExpr>>(),
                                (0..dims.len())
                                    .map(|x| { Load::make(&format!("%{}.s", id), x).into() })
                                    .collect::<Vec<PrimeExpr>>(),
                                Variable::make(&format!("%{}_val", stage.out_id))
                            )
                        );
                        stage.bodys.push(body);
                        let stage = Stage {
                            dims,
                            bodys: stage.bodys.clone(),
                            id,
                            out_id: stage.out_id,
                            dtype: stage.dtype,
                            begins: stage.begins.clone(),
                            steps: stage.steps.clone(),
                            axes: stage.axes.clone(),
                        };
                        Body::Stage(stage)
                    } else {
                        panic!("input is not a stage");
                    }
                } else {
                    if let Body::Stage(stage) = &inputs[0] {
                        let mut stage = stage.clone();
                        let dims = (0..shape.len())
                            .map(|x|
                                IterVar::new(0i64, shape[x].clone(), 1i64, &format!("ax{}", x))
                            )
                            .collect::<Vec<IterVar>>();
                        stage.broadcast_new_dims(
                            &(0..shape.len())
                                .map(|x|
                                    Load::make(Variable::make(&format!("%{}.s", id)), x).into()
                                )
                                .collect::<Vec<PrimeExpr>>(),
                            &(0..shape.len())
                                .map(|x| Variable::make(&format!("ax{}", x)).into())
                                .collect::<Vec<PrimeExpr>>()
                        );
                        let body = Body::Stmt(
                            Stmt::LetStmt(
                                LetStmt::make(
                                    &Variable::make(&format!("%{}_val", id)),
                                    Variable::make(&format!("%{}_val", stage.out_id)),
                                    false,
                                    Stmt::None
                                )
                            ).into()
                        );
                        stage.bodys.push(body);
                        let stage = Stage {
                            dims,
                            bodys: stage.bodys.clone(),
                            id,
                            out_id: stage.out_id,
                            dtype: stage.dtype,
                            begins: stage.begins.clone(),
                            steps: stage.steps.clone(),
                            axes: stage.axes.clone(),
                        };
                        Body::Stage(stage)
                    } else {
                        panic!("input is not a stage");
                    }
                }
            }),
        };
        self.nodes.borrow_mut().insert(id, tensor.clone());
        tensor
    }

    #[track_caller]
    pub fn slice(
        &mut self,
        a: &Tensor,
        selections: &[
            (&dyn ToPrimeExpr /*begin */, &dyn ToPrimeExpr /*end */, &dyn ToPrimeExpr /*step */)
        ]
    ) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1i64));
        let zero = PrimeExpr::Int(Int::make(Dtype::I64, 0i64));
        let selections = selections
            .iter()
            .map(|(begin, end, step)| {
                let mut const_fold = ConstFold::new();
                (
                    const_fold.const_fold(begin.to_prime_expr()),
                    const_fold.const_fold(end.to_prime_expr()),
                    const_fold.const_fold(step.to_prime_expr()),
                )
            })
            .collect::<Vec<(PrimeExpr, PrimeExpr, PrimeExpr)>>();
        let new_shape = Arc::new(
            selections
                .iter()
                .map(|(begin, end, step)| {
                    if begin == &zero && step == &one {
                        end.clone()
                    } else if step == &one {
                        end.clone() - begin.clone()
                    } else {
                        if end.is_int() && begin.is_int() && step.is_int() {
                            let end = end.to_int().unwrap();
                            let begin = begin.to_int().unwrap();
                            let step = step.to_int().unwrap();
                            let sub = end.value() - begin.value();
                            ((sub + step.value() - 1) / step.value()).into()
                        } else {
                            let mut const_fold = ConstFold::new();
                            Call::make(
                                "ceil",
                                &[
                                    const_fold.const_fold(
                                        (end.clone() - begin.clone()) / step.clone()
                                    ),
                                ]
                            ).into()
                        }
                    }
                })
                .map(|x| { x })
                .collect::<Vec<_>>()
        );
        let slice = Arc::new(selections.clone());
        let shape = new_shape.clone();
        let ret = Tensor {
            shape: new_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            dtype: a.dtype.clone(),
            id,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                slice_strides_cal(prev_fn[0].clone())
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                if is_output {
                    if let Body::Stage(stage) = &inputs[0] {
                        let stage = stage.clone();
                        let dims = (0..stage.dims.len())
                            .map(|idx|
                                IterVar::new(0i64, shape[idx].clone(), 1i64, &format!("ax{}", idx))
                            )
                            .collect::<Vec<IterVar>>();
                        let begins = slice
                            .iter()
                            .map(|(begin, _, _)| begin.clone())
                            .collect();
                        let steps = slice
                            .iter()
                            .map(|(_, _, step)| step.clone())
                            .collect();
                        let mut slice_visitor = SliceVisitor::new(begins, steps);

                        let mut bodys = stage.bodys
                            .iter()
                            .map(|x| {
                                let mut x = x.clone();
                                slice_visitor.set_expr(PrimeExpr::None);
                                slice_visitor.set_stmt(Stmt::None);
                                x.accept_mutate(&mut slice_visitor);
                                x
                            })
                            .collect::<Vec<Body>>();
                        let store_body = Body::Stmt(
                            store_with_dims(
                                format!("%{}", id),
                                dims
                                    .iter()
                                    .map(|x| x.var().to_prime_expr())
                                    .collect::<Vec<PrimeExpr>>(),
                                (0..dims.len())
                                    .map(|x| { Load::make(&format!("%{}.s", id), x).into() })
                                    .collect::<Vec<PrimeExpr>>(),
                                Variable::make(&format!("%{}_val", stage.out_id))
                            )
                        );
                        bodys.push(store_body);
                        let stage = Stage {
                            dims,
                            bodys,
                            id,
                            out_id: id,
                            dtype: stage.dtype,
                            begins: stage.begins.clone(),
                            steps: stage.steps.clone(),
                            axes: stage.axes.clone(),
                        };
                        Body::Stage(stage)
                    } else {
                        panic!("input is not a stage");
                    }
                } else {
                    if let Body::Stage(stage) = &inputs[0] {
                        let stage = stage.clone();
                        let dims = (0..stage.dims.len())
                            .map(|idx|
                                IterVar::new(0i64, shape[idx].clone(), 1i64, &format!("ax{}", idx))
                            )
                            .collect::<Vec<IterVar>>();
                        let begins = slice
                            .iter()
                            .map(|(begin, _, _)| begin.clone())
                            .collect();
                        let steps = slice
                            .iter()
                            .map(|(_, _, step)| step.clone())
                            .collect();
                        let mut slice_visitor = SliceVisitor::new(begins, steps);
                        let mut bodys = stage.bodys
                            .iter()
                            .map(|x| {
                                let mut x = x.clone();
                                slice_visitor.set_expr(PrimeExpr::None);
                                slice_visitor.set_stmt(Stmt::None);
                                x.accept_mutate(&mut slice_visitor);
                                x
                            })
                            .collect::<Vec<Body>>();
                        let let_body = Body::Stmt(
                            Stmt::LetStmt(
                                LetStmt::make(
                                    &Variable::make(&format!("%{}_val", id)),
                                    Variable::make(&format!("%{}_val", stage.out_id)),
                                    false,
                                    Stmt::None
                                )
                            ).into()
                        );
                        bodys.push(let_body);
                        let stage = Stage {
                            dims,
                            bodys,
                            id,
                            out_id: id,
                            dtype: stage.dtype,
                            begins: stage.begins.clone(),
                            steps: stage.steps.clone(),
                            axes: stage.axes.clone(),
                        };
                        Body::Stage(stage)
                    } else {
                        panic!("input is not a stage");
                    }
                }
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
