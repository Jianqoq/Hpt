use std::{ collections::{ HashMap, HashSet }, sync::Arc };

use tensor_common::{
    shape::Shape,
    shape_utils::{ is_reshape_possible, try_pad_shape },
    slice::{ slice_process, Slice },
    strides_utils::{ preprocess_strides, shape_to_strides },
};
use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::{ Call, Int, Load },
        inplace_store_stmt::InplaceAdd,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        store_stmt::StoreStmt,
        tensor_load::TensorLoad,
        variable::Variable,
    },
    iter_var::IterVar,
    te::{ hstrides::HStrides, idx_evaluator::IdxEvaluator, stages::{ Body, ReduceStage, Stage } },
    to_prim_expr::ToPrimeExpr,
};

use super::{
    operation::Operation,
    rc_mut::RcMut,
    schedule::Schedule,
    srg_node::SrgNode,
    tensor::Tensor,
};

pub struct Srg {
    pub(crate) nodes: HashMap<usize, SrgNode>,
    pub(crate) tensors: RcMut<HashMap<usize, Tensor>>,
}

impl Srg {
    pub fn create_strides_cal(&mut self, sorted: &[usize]) {
        for id in sorted {
            let node_op = self.nodes[&id].op.clone();
            let inputs = self.nodes[&id].inputs.clone();
            let span_location = self.nodes[&id].span.clone();
            if inputs.len() == 0 {
                let node = self.nodes.get_mut(&id).unwrap();
                let node_shape = node.shape.clone();
                let func = Arc::new(move |map: &HashMap<String, i64>| {
                    let real_shape = node_shape
                        .iter()
                        .map(|x| { IdxEvaluator::new(map).eval(x) })
                        .collect::<Vec<i64>>();
                    let hstrides = HStrides {
                        strides: shape_to_strides(&real_shape).inner().clone(),
                        reduced_dim: 0,
                        offset: 0,
                    };
                    vec![hstrides]
                });
                node.strides_cal = func;
            } else {
                match node_op {
                    Operation::Reshape(new_shape) => {
                        let (prev_func, prev_shape) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].strides_cal.clone(),
                                self.nodes[&node.inputs[0]].shape.clone(),
                            )
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let prev_strides = prev_func(map);
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
                                    assert!(new.len() == new_shape.len() + i.reduced_dim);
                                    let new = HStrides {
                                        strides: new,
                                        reduced_dim: i.reduced_dim,
                                        offset: i.offset,
                                    };
                                    ret.push(new);
                                } else {
                                    panic!("Reshape not possible, {}", span_location);
                                }
                            }
                            ret
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.strides_cal = func;
                    }
                    Operation::Transpose(axes) => {
                        let prev_func = {
                            let node = &self.nodes[&id];
                            self.nodes[&node.inputs[0]].strides_cal.clone()
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let prev_strides = prev_func(map);
                            let mut ret = vec![];
                            for strides in prev_strides.into_iter() {
                                let masked =
                                    &strides[..strides.strides.len() - strides.reduced_dim];
                                let mut new_strides = vec![];
                                for i in axes.iter() {
                                    new_strides.push(masked[*i]);
                                }
                                for i in strides[
                                    strides.strides.len() - strides.reduced_dim..
                                ].iter() {
                                    new_strides.push(*i);
                                }
                                assert!(new_strides.len() == strides.strides.len());
                                let new = HStrides {
                                    strides: new_strides,
                                    reduced_dim: strides.reduced_dim,
                                    offset: strides.offset,
                                };
                                ret.push(new);
                            }
                            ret
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.strides_cal = func;
                    }
                    Operation::Sum(axes, _) => {
                        let prev_func = {
                            let node = &self.nodes[&id];
                            self.nodes[&node.inputs[0]].strides_cal.clone()
                        };
                        let axes_len = axes.len();
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let prev_strides = prev_func(map);
                            let mut ret = vec![];
                            for strides in prev_strides.into_iter() {
                                let masked =
                                    &strides[..strides.strides.len() - strides.reduced_dim];
                                let mut new_strides = vec![];
                                for i in 0..masked.len() {
                                    if axes.contains(&i) {
                                        continue;
                                    }
                                    new_strides.push(masked[i]);
                                }
                                for i in 0..masked.len() {
                                    if axes.contains(&i) {
                                        new_strides.push(masked[i]);
                                    }
                                }
                                for i in strides[
                                    strides.strides.len() - strides.reduced_dim..
                                ].iter() {
                                    new_strides.push(*i);
                                }
                                assert!(new_strides.len() == strides.strides.len());
                                let new = HStrides {
                                    strides: new_strides,
                                    reduced_dim: strides.reduced_dim + axes_len,
                                    offset: strides.offset,
                                };
                                ret.push(new);
                            }
                            ret
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.strides_cal = func;
                    }
                    Operation::Sin => {
                        let prev_func = {
                            let node = &self.nodes[&id];
                            self.nodes[&node.inputs[0]].strides_cal.clone()
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| { prev_func(map) });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.strides_cal = func;
                    }
                    Operation::Add => {
                        let lhs_prev_func = {
                            let node = &self.nodes[&id];
                            self.nodes[&node.inputs[0]].strides_cal.clone()
                        };
                        let rhs_prev_func = {
                            let node = &self.nodes[&id];
                            self.nodes[&node.inputs[1]].strides_cal.clone()
                        };
                        let (lhs_shape, rhs_shape) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].shape.clone(),
                                self.nodes[&node.inputs[1]].shape.clone(),
                            )
                        };
                        let res_shape = {
                            let node = &self.nodes[&id];
                            node.shape.clone()
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let lhs_strides = lhs_prev_func(map);
                            let rhs_strides = rhs_prev_func(map);
                            assert_eq!(lhs_strides.len(), rhs_strides.len());

                            let lhs_real_shape = lhs_shape
                                .iter()
                                .map(|x| { IdxEvaluator::new(map).eval(x) })
                                .collect::<Vec<i64>>();

                            let rhs_real_shape = rhs_shape
                                .iter()
                                .map(|x| { IdxEvaluator::new(map).eval(x) })
                                .collect::<Vec<i64>>();

                            let res_real_shape = res_shape
                                .iter()
                                .map(|x| { IdxEvaluator::new(map).eval(x) })
                                .collect::<Vec<i64>>();

                            let mut lhs_strides_vec = vec![];
                            for strides in lhs_strides.into_iter() {
                                let masked =
                                    &strides[..strides.strides.len() - strides.reduced_dim];
                                assert!(masked.len() == lhs_real_shape.len());
                                let padded = try_pad_shape(&lhs_real_shape, res_real_shape.len());
                                let new = preprocess_strides::<_, _, i64>(&padded, masked);
                                let mut new_strides = vec![];
                                for i in new.iter() {
                                    new_strides.push(*i);
                                }
                                for i in strides[
                                    strides.strides.len() - strides.reduced_dim..
                                ].iter() {
                                    new_strides.push(*i);
                                }
                                let new = HStrides {
                                    strides: new_strides,
                                    reduced_dim: strides.reduced_dim,
                                    offset: strides.offset,
                                };
                                lhs_strides_vec.push(new);
                            }
                            let mut rhs_strides_vec = vec![];
                            for strides in rhs_strides.into_iter() {
                                let masked =
                                    &strides[..strides.strides.len() - strides.reduced_dim];
                                assert!(masked.len() == rhs_real_shape.len());
                                let padded = try_pad_shape(&rhs_real_shape, res_real_shape.len());
                                let new = preprocess_strides::<_, _, i64>(&padded, masked);
                                let mut new_strides = vec![];
                                for i in new.iter() {
                                    new_strides.push(*i);
                                }
                                for i in strides[
                                    strides.strides.len() - strides.reduced_dim..
                                ].iter() {
                                    new_strides.push(*i);
                                }
                                let new = HStrides {
                                    strides: new_strides,
                                    reduced_dim: strides.reduced_dim,
                                    offset: strides.offset,
                                };
                                rhs_strides_vec.push(new);
                            }
                            lhs_strides_vec.extend(rhs_strides_vec);
                            lhs_strides_vec
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.strides_cal = func;
                    }
                    Operation::Slice(selections) => {
                        let strides_cal = {
                            let node = &self.nodes[&id];
                            assert!(node.inputs.len() == 1);
                            assert!(self.nodes[&node.inputs[0]].inputs.len() == 0);
                            self.nodes[&node.inputs[0]].strides_cal.clone()
                        };
                        let shape = {
                            let node = &self.nodes[&id];
                            node.shape.clone()
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let selections = selections
                                .iter()
                                .map(|(start, end, step)| (
                                    IdxEvaluator::new(map).eval(start),
                                    IdxEvaluator::new(map).eval(end),
                                    IdxEvaluator::new(map).eval(step),
                                ))
                                .collect::<Vec<(i64, i64, i64)>>();
                            let real_shape = shape
                                .iter()
                                .map(|x| { IdxEvaluator::new(map).eval(x) })
                                .collect::<Vec<i64>>();
                            let real_strides = shape_to_strides(&real_shape); // we always aloccate memory before we slice
                            let (_, strides, offset) = slice_process(
                                real_shape,
                                real_strides.inner().clone(),
                                &selections
                                    .into_iter()
                                    .map(|(x, y, z)| Slice::StepByRangeFromTo((x, y, z)))
                                    .collect::<Vec<Slice>>(),
                                1
                            ).expect("slice process failed");
                            let mut strideses = strides_cal(map);
                            assert!(strideses.len() == 1);
                            let hstrides = HStrides {
                                strides,
                                reduced_dim: 0,
                                offset,
                            };
                            strideses.push(hstrides);
                            strideses
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.strides_cal = func;
                    }
                    Operation::None => unreachable!(),
                }
            }
        }
    }

    pub fn create_schedule(&self, sorted: &[usize]) -> Schedule {
        let mut declared_vars = HashSet::new();
        let mut qa = HashMap::new();
        for id in sorted {
            let node = &self.nodes[id];
            if node.inputs.len() == 0 {
                let body = Body::Stmt(
                    Stmt::LetStmt(
                        LetStmt::make(
                            &Variable::make(&format!("%{}_val", node.id)),
                            TensorLoad {
                                var: Variable::make(&format!("%{}", node.id)).into(),
                                axes: (0..node.shape.len())
                                    .map(|x| Variable::make(&format!("ax{}", x)).into())
                                    .collect::<Vec<PrimeExpr>>()
                                    .into(),
                                strides: (0..node.shape.len())
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
                            Stmt::None
                        )
                    ).into()
                );
                let stage = Stage {
                    dims: (0..node.shape.len())
                        .map(|x|
                            IterVar::new(0i64, node.shape[x].clone(), 1i64, &format!("ax{}", x))
                        )
                        .collect(),
                    bodys: vec![body],
                    id: *id,
                };
                qa.insert(*id, (Body::Stage(stage), false));
                declared_vars.insert(format!("%{}_val", node.id));
            } else {
                match &node.op {
                    Operation::Reshape(reshape) => {
                        let (input, _) = &qa[&node.inputs[0]];
                        if node.is_output() {
                            if let Body::Stage(stage) = input {
                                let mut stage = stage.clone();
                                let dims = (0..reshape.len())
                                    .map(|x|
                                        IterVar::new(
                                            0i64,
                                            reshape[x].clone(),
                                            1i64,
                                            &format!("ax{}", x)
                                        )
                                    )
                                    .collect::<Vec<IterVar>>();
                                stage.broadcast_new_dims(
                                    &(0..reshape.len())
                                        .map(|x|
                                            Load::make(
                                                Variable::make(&format!("%{}.s", id)),
                                                x
                                            ).into()
                                        )
                                        .collect::<Vec<PrimeExpr>>(),
                                    &(0..reshape.len())
                                        .map(|x| Variable::make(&format!("ax{}", x)).into())
                                        .collect::<Vec<PrimeExpr>>()
                                );
                                let body = Body::Stmt(
                                    Stmt::StoreStmt(
                                        StoreStmt::make(
                                            &Variable::make(&format!("%{}", id)),
                                            dims
                                                .iter()
                                                .enumerate()
                                                .map(
                                                    |(idx, x)|
                                                        x.var().to_prime_expr() *
                                                        Load::make(
                                                            &format!("%{}.s", id),
                                                            idx
                                                        ).into()
                                                )
                                                .reduce(|acc, x| acc + x)
                                                .unwrap(),
                                            Variable::make(&format!("%{}_val", node.inputs[0]))
                                        )
                                    )
                                );
                                stage.bodys.push(body);
                                let stage = Stage {
                                    dims,
                                    bodys: stage.bodys.clone(),
                                    id: *id,
                                };
                                qa.insert(*id, (Body::Stage(stage), true));
                            } else {
                                panic!("input is not a stage");
                            }
                        } else {
                            if let Body::Stage(stage) = input {
                                let mut stage = stage.clone();
                                let dims = (0..reshape.len())
                                    .map(|x|
                                        IterVar::new(
                                            0i64,
                                            reshape[x].clone(),
                                            1i64,
                                            &format!("ax{}", x)
                                        )
                                    )
                                    .collect::<Vec<IterVar>>();
                                stage.broadcast_new_dims(
                                    &(0..reshape.len())
                                        .map(|x|
                                            Load::make(
                                                Variable::make(&format!("%{}.s", id)),
                                                x
                                            ).into()
                                        )
                                        .collect::<Vec<PrimeExpr>>(),
                                    &(0..reshape.len())
                                        .map(|x| Variable::make(&format!("ax{}", x)).into())
                                        .collect::<Vec<PrimeExpr>>()
                                );
                                let body = Body::Stmt(
                                    Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_val", node.id)),
                                            Variable::make(&format!("%{}_val", node.inputs[0])),
                                            Stmt::None
                                        )
                                    ).into()
                                );
                                stage.bodys.push(body);
                                let stage = Stage {
                                    dims,
                                    bodys: stage.bodys.clone(),
                                    id: *id,
                                };
                                qa.insert(*id, (Body::Stage(stage), false));
                            } else {
                                panic!("input is not a stage");
                            }
                        }
                    }
                    Operation::Transpose(_) => {}
                    Operation::Sum(axes, init) => {
                        let (input, _) = qa.get(&node.inputs[0]).unwrap();
                        if node.is_output() {
                            if let Body::Stage(stage) = input {
                                let mut stage = stage.clone();
                                let mut dims = (0..node.shape.len())
                                    .map(|x| Variable::make(&format!("ax{}", x)).into())
                                    .collect::<Vec<PrimeExpr>>();
                                let mut red_dims = vec![];
                                for i in axes.iter() {
                                    red_dims.push(
                                        Variable::make(&format!("{}red{}", id, i)).into()
                                    );
                                }
                                dims.extend(red_dims);
                                stage.broadcast_new_dims(
                                    &(0..self.nodes[&node.inputs[0]].shape.len())
                                        .map(|x|
                                            Load::make(
                                                Variable::make(&format!("%{}.s", id)),
                                                x
                                            ).into()
                                        )
                                        .collect::<Vec<PrimeExpr>>(),
                                    &dims
                                );
                                let mut bodys = stage.bodys.clone();
                                bodys.push(
                                    Body::Stmt(
                                        Stmt::InplaceAdd(
                                            InplaceAdd::make(
                                                &Variable::make(&format!("%{}_val", node.id)),
                                                Variable::make(&format!("%{}_val", node.inputs[0]))
                                            )
                                        )
                                    )
                                );
                                stage.dims = (0..self.nodes[&node.inputs[0]].shape.len())
                                    .filter(|x| !axes.contains(x))
                                    .enumerate()
                                    .map(|(idx, x)|
                                        IterVar::new(
                                            0i64,
                                            self.nodes[&node.inputs[0]].shape[x].clone(),
                                            1i64,
                                            &format!("ax{}", idx)
                                        )
                                    )
                                    .collect();
                                let all_parent_dims = &stage.dims;
                                let red_axes = axes
                                    .iter()
                                    .map(|x|
                                        IterVar::new(
                                            0i64,
                                            self.nodes[&node.inputs[0]].shape[*x].clone(),
                                            1i64,
                                            &format!("{}red{}", id, x)
                                        )
                                    )
                                    .collect::<Vec<IterVar>>();
                                let reduce_stage = ReduceStage {
                                    dims: red_axes.clone(),
                                    bodys,
                                    id: *id,
                                    inits: vec![
                                        Body::Stmt(
                                            Stmt::LetStmt(
                                                LetStmt::make(
                                                    &Variable::make(&format!("%{}_val", node.id)),
                                                    init.clone(),
                                                    Stmt::None
                                                )
                                            ).into()
                                        )
                                    ],
                                    posts: vec![
                                        Body::Stmt(
                                            Stmt::StoreStmt(
                                                StoreStmt::make(
                                                    &Variable::make(&format!("%{}", id)),
                                                    all_parent_dims
                                                        .iter()
                                                        .enumerate()
                                                        .map(
                                                            |(idx, x)|
                                                                PrimeExpr::Variable(
                                                                    x.var().clone()
                                                                ) *
                                                                Load::make(
                                                                    &format!("%{}.s", id),
                                                                    idx
                                                                ).into()
                                                        )
                                                        .reduce(|acc, x| acc + x)
                                                        .unwrap_or((0i64).into()),
                                                    Variable::make(&format!("%{}_val", id))
                                                )
                                            )
                                        )
                                    ],
                                    input: node.inputs[0],
                                };
                                stage.bodys = vec![Body::ReduceStage(reduce_stage)];
                                qa.insert(*id, (Body::Stage(stage), true));
                            } else {
                                panic!("input is not a stage");
                            }
                        } else {
                            if let Body::Stage(stage) = input {
                                let mut stage = stage.clone();
                                let mut dims = (0..node.shape.len())
                                    .map(|x| Variable::make(&format!("ax{}", x)).into())
                                    .collect::<Vec<PrimeExpr>>();
                                let mut red_dims = vec![];
                                for i in axes.iter() {
                                    red_dims.push(
                                        Variable::make(&format!("{}red{}", id, i)).into()
                                    );
                                }
                                dims.extend(red_dims);
                                stage.broadcast_new_dims(
                                    &(0..self.nodes[&node.inputs[0]].shape.len())
                                        .map(|x|
                                            Load::make(
                                                Variable::make(&format!("%{}.s", id)),
                                                x
                                            ).into()
                                        )
                                        .collect::<Vec<PrimeExpr>>(),
                                    &dims
                                );
                                let mut bodys = stage.bodys.clone();
                                bodys.push(
                                    Body::Stmt(
                                        Stmt::InplaceAdd(
                                            InplaceAdd::make(
                                                &Variable::make(&format!("%{}_val", node.id)),
                                                Variable::make(&format!("%{}_val", node.inputs[0]))
                                            )
                                        )
                                    )
                                );
                                stage.dims = (0..self.nodes[&node.inputs[0]].shape.len())
                                    .filter(|x| !axes.contains(x))
                                    .enumerate()
                                    .map(|(idx, x)|
                                        IterVar::new(
                                            0i64,
                                            self.nodes[&node.inputs[0]].shape[x].clone(),
                                            1i64,
                                            &format!("ax{}", idx)
                                        )
                                    )
                                    .collect();
                                let red_axes = axes
                                    .iter()
                                    .map(|x|
                                        IterVar::new(
                                            0i64,
                                            self.nodes[&node.inputs[0]].shape[*x].clone(),
                                            1i64,
                                            &format!("{}red{}", id, x)
                                        )
                                    )
                                    .collect::<Vec<IterVar>>();
                                let reduce_stage = ReduceStage {
                                    dims: red_axes.clone(),
                                    bodys,
                                    id: *id,
                                    inits: vec![
                                        Body::Stmt(
                                            Stmt::LetStmt(
                                                LetStmt::make(
                                                    &Variable::make(&format!("%{}_val", node.id)),
                                                    init.clone(),
                                                    Stmt::None
                                                )
                                            ).into()
                                        )
                                    ],
                                    posts: vec![],
                                    input: node.inputs[0],
                                };
                                stage.bodys = vec![Body::ReduceStage(reduce_stage)];
                                qa.insert(*id, (Body::Stage(stage), false));
                            } else {
                                panic!("input is not a stage");
                            }
                        }
                    }
                    Operation::Sin => {
                        let (input, _) = qa.get(&node.inputs[0]).unwrap();
                        if node.is_output() {
                            if let Body::Stage(stage) = input {
                                let mut stage = stage.clone();
                                let body = Body::Stmt(
                                    Stmt::StoreStmt(
                                        StoreStmt::make(
                                            &Variable::make(&format!("%{}", id)),
                                            stage.dims
                                                .iter()
                                                .enumerate()
                                                .map(
                                                    |(idx, x)|
                                                        x.var().to_prime_expr() *
                                                        Load::make(
                                                            &format!("%{}.s", id),
                                                            idx
                                                        ).into()
                                                )
                                                .reduce(|acc, x| acc + x)
                                                .unwrap(),
                                            Variable::make(&format!("%{}_val", node.inputs[0]))
                                        )
                                    )
                                );
                                stage.bodys.push(body);
                                let stage = Stage {
                                    dims: stage.dims.clone(),
                                    bodys: stage.bodys.clone(),
                                    id: *id,
                                };
                                qa.insert(*id, (Body::Stage(stage), true));
                            } else {
                                panic!("input is not a stage");
                            }
                        } else {
                            if let Body::Stage(stage) = input {
                                let body = Body::Stmt(
                                    Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_val", node.id)),
                                            Call::make(
                                                "sin",
                                                &[
                                                    Load::make(
                                                        Variable::make(
                                                            &format!("%{}_val", node.inputs[0])
                                                        ),
                                                        0
                                                    ),
                                                ]
                                            ),
                                            Stmt::None
                                        )
                                    ).into()
                                );
                                let mut stage_bodys = stage.bodys.clone();
                                stage_bodys.push(body);
                                let stage = Stage {
                                    dims: stage.dims.clone(),
                                    bodys: stage_bodys,
                                    id: *id,
                                };
                                qa.insert(*id, (Body::Stage(stage), false));
                            } else {
                                panic!("input is not a stage");
                            }
                        }
                    }
                    Operation::Add => {
                        let (lhs, _) = &qa[&node.inputs[0]];
                        let (rhs, _) = &qa[&node.inputs[1]];
                        let dims = self.nodes[id].shape
                            .iter()
                            .enumerate()
                            .map(|(idx, x)|
                                IterVar::new(0i64, x.clone(), 1i64, &format!("ax{}", idx))
                            )
                            .collect::<Vec<IterVar>>();
                        if node.is_output() {
                            match (lhs, rhs) {
                                (Body::Stage(lhs), Body::Stage(rhs)) => {
                                    let mut lhs_bodys = lhs.bodys.clone();
                                    let rhs_bodys = rhs.bodys.clone();
                                    lhs_bodys.extend(rhs_bodys);
                                    let sotre_add = Stmt::StoreStmt(
                                        StoreStmt::make(
                                            &Variable::make(&format!("%{}", id)),
                                            dims
                                                .iter()
                                                .enumerate()
                                                .map(
                                                    |(idx, x)|
                                                        x.var().to_prime_expr() *
                                                        Load::make(
                                                            &format!("%{}.s", id),
                                                            idx
                                                        ).into()
                                                )
                                                .reduce(|acc, x| acc + x)
                                                .unwrap(),
                                            Variable::make(&format!("%{}_val", node.inputs[0])) +
                                                Variable::make(&format!("%{}_val", node.inputs[1]))
                                        )
                                    );
                                    lhs_bodys.push(Body::Stmt(sotre_add));
                                    let stage = Stage {
                                        dims,
                                        bodys: lhs_bodys,
                                        id: *id,
                                    };
                                    qa.insert(*id, (Body::Stage(stage), true));
                                }
                                _ => panic!("input is not a stage"),
                            }
                        } else {
                            match (lhs, rhs) {
                                (Body::Stage(lhs), Body::Stage(rhs)) => {
                                    let mut lhs_bodys = lhs.bodys.clone();
                                    let rhs_bodys = rhs.bodys.clone();
                                    lhs_bodys.extend(rhs_bodys);
                                    let add = Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_val", id)),
                                            Variable::make(&format!("%{}_val", node.inputs[0])) +
                                                Variable::make(&format!("%{}_val", node.inputs[1])),
                                            Stmt::None
                                        )
                                    );
                                    lhs_bodys.push(Body::Stmt(add));
                                    let stage = Stage {
                                        dims,
                                        bodys: lhs_bodys,
                                        id: *id,
                                    };
                                    qa.insert(*id, (Body::Stage(stage), false));
                                }
                                _ => panic!("input is not a stage"),
                            }
                        }
                    }
                    Operation::Slice(slice) => {
                        let (input, _) = qa.get(&node.inputs[0]).unwrap();
                        if node.is_output() {
                            if let Body::Stage(stage) = input {
                                let stage = stage.clone();
                                let dims = (0..stage.dims.len())
                                    .zip(slice.iter())
                                    .map(|(idx, (start, end, step))|
                                        IterVar::new(
                                            0i64,
                                            (end -
                                                start +
                                                step -
                                                PrimeExpr::Int(Int::make(Dtype::I64, 1))) /
                                                step.clone(),
                                            1i64,
                                            &format!("ax{}", idx)
                                        )
                                    )
                                    .collect::<Vec<IterVar>>();
                                let offset_body = Body::Stmt(
                                    Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_ptr", node.id)),
                                            PrimeExpr::Variable(
                                                Variable::make(&format!("%{}", node.inputs[0]))
                                            ) +
                                                PrimeExpr::Variable(
                                                    Variable::make(&format!("%{}_offset", id))
                                                ),
                                            Stmt::None
                                        )
                                    )
                                );
                                let body = Body::Stmt(
                                    Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_val", id)),
                                            TensorLoad {
                                                var: Variable::make(
                                                    &format!("%{}_ptr", node.id)
                                                ).into(),
                                                axes: (0..node.shape.len())
                                                    .map(|x|
                                                        Variable::make(&format!("ax{}", x)).into()
                                                    )
                                                    .collect::<Vec<PrimeExpr>>()
                                                    .into(),
                                                strides: (0..node.shape.len())
                                                    .map(|idx|
                                                        Load::make(
                                                            Variable::make(
                                                                &format!("%{}.s", node.inputs[0])
                                                            ),
                                                            idx
                                                        ).into()
                                                    )
                                                    .collect::<Vec<PrimeExpr>>()
                                                    .into(),
                                                hints: vec![].into(),
                                            },
                                            Stmt::None
                                        )
                                    ).into()
                                );
                                let store_body = Body::Stmt(
                                    Stmt::StoreStmt(
                                        StoreStmt::make(
                                            &Variable::make(&format!("%{}", id)),
                                            dims
                                                .iter()
                                                .enumerate()
                                                .map(
                                                    |(idx, x)|
                                                        x.var().to_prime_expr() *
                                                        Load::make(
                                                            &format!("%{}.s", id),
                                                            idx
                                                        ).into()
                                                )
                                                .reduce(|acc, x| acc + x)
                                                .unwrap(),
                                            Variable::make(&format!("%{}_val", node.inputs[0]))
                                        )
                                    )
                                );
                                let stage = Stage {
                                    dims,
                                    bodys: vec![offset_body, body, store_body],
                                    id: *id,
                                };
                                qa.insert(*id, (Body::Stage(stage), true));
                            } else {
                                panic!("input is not a stage");
                            }
                        } else {
                            if let Body::Stage(stage) = input {
                                let stage = stage.clone();
                                let dims = (0..stage.dims.len())
                                    .zip(slice.iter())
                                    .map(|(idx, (start, end, step))|
                                        IterVar::new(
                                            0i64,
                                            (end -
                                                start +
                                                step -
                                                PrimeExpr::Int(Int::make(Dtype::I64, 1))) /
                                                step.clone(),
                                            1i64,
                                            &format!("ax{}", idx)
                                        )
                                    )
                                    .collect::<Vec<IterVar>>();
                                let offset_body = Body::Stmt(
                                    Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_ptr", node.id)),
                                            PrimeExpr::Variable(
                                                Variable::make(&format!("%{}", node.inputs[0]))
                                            ) +
                                                PrimeExpr::Variable(
                                                    Variable::make(&format!("%{}_offset", id))
                                                ),
                                            Stmt::None
                                        )
                                    )
                                );
                                let body = Body::Stmt(
                                    Stmt::LetStmt(
                                        LetStmt::make(
                                            &Variable::make(&format!("%{}_val", id)),
                                            TensorLoad {
                                                var: Variable::make(
                                                    &format!("%{}_ptr", node.id)
                                                ).into(),
                                                axes: (0..node.shape.len())
                                                    .map(|x|
                                                        Variable::make(&format!("ax{}", x)).into()
                                                    )
                                                    .collect::<Vec<PrimeExpr>>()
                                                    .into(),
                                                strides: (0..node.shape.len())
                                                    .map(|idx|
                                                        Load::make(
                                                            Variable::make(
                                                                &format!("%{}.s", node.inputs[0])
                                                            ),
                                                            idx
                                                        ).into()
                                                    )
                                                    .collect::<Vec<PrimeExpr>>()
                                                    .into(),
                                                hints: vec![].into(),
                                            },
                                            Stmt::None
                                        )
                                    ).into()
                                );
                                let stage = Stage {
                                    dims,
                                    bodys: vec![offset_body, body],
                                    id: *id,
                                };
                                qa.insert(*id, (Body::Stage(stage), false));
                            } else {
                                panic!("input is not a stage");
                            }
                        }
                    }
                    Operation::None => {}
                }
            }
        }

        let strides_cal = self.nodes
            .values()
            .filter_map(|node| {
                if node.is_output() { Some(node.strides_cal.clone()) } else { None }
            })
            .last()
            .unwrap();
        Schedule { qa, nodes: self.tensors.clone(), strides_cal }
    }
}

#[cfg(test)]
mod tests {
    use std::{ collections::HashMap, sync::Arc };

    use tensor_types::dtype::Dtype;

    use crate::{
        halide::{ exprs::Int, prime_expr::PrimeExpr },
        te::{ context::Context, srg_node::SrgNode },
    };

    use super::Srg;

    #[test]
    fn test_srg_create_strides_cal() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.placeholder(&[&1i64], Dtype::F32);
        let c = ctx.add(&a, &b);
        let d = ctx.sum(&c, &0i64, &[2]);
        let e = ctx.reshape(&d, &[&m, &n, &1i64]);
        let f = ctx.add(&c, &e);
        let g = ctx.sin(&f);
        let h = ctx.sum(&g, &0i64, &[2]);
        let i = ctx.reshape(&h, &[&m, &n, &1i64]);
        let j = ctx.add(&g, &i);
        let order = [a.id, b.id, c.id, d.id, e.id, f.id, g.id, h.id, i.id, j.id];

        let mut nodes = HashMap::new();
        for (id, node) in ctx.nodes.borrow().iter() {
            let srg_node = SrgNode {
                id: *id,
                shape: node.shape.clone(),
                inputs: node.inputs.clone(),
                outputs: Arc::new(
                    ctx.nodes
                        .borrow()
                        .iter()
                        .filter_map(|(k, v)| {
                            if v.inputs.contains(id) { Some(*k) } else { None }
                        })
                        .collect()
                ),
                op: node.op.clone(),
                strides_cal: Arc::new(|_| vec![]),
                span: node.span,
            };
            nodes.insert(*id, srg_node);
        }
        let mut srg = Srg {
            nodes,
            tensors: ctx.nodes.clone(),
        };
        srg.create_strides_cal(&order);

        let mut var_map = HashMap::new();
        var_map.insert("m".to_string(), 1);
        var_map.insert("n".to_string(), 8);
        var_map.insert("o".to_string(), 8);

        let node = &srg.nodes[order.last().unwrap()];
        let strides = (node.strides_cal)(&var_map);
        println!("{:?}", strides);
    }

    #[test]
    fn test_reshape_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.reshape(&a, &[&m, &n, &o, &1i64]);
        let order = [a.id, b.id];

        let mut nodes = HashMap::new();
        for (id, node) in ctx.nodes.borrow().iter() {
            let srg_node = SrgNode {
                id: *id,
                shape: node.shape.clone(),
                inputs: node.inputs.clone(),
                outputs: Arc::new(
                    ctx.nodes
                        .borrow()
                        .iter()
                        .filter_map(|(k, v)| {
                            if v.inputs.contains(id) { Some(*k) } else { None }
                        })
                        .collect()
                ),
                op: node.op.clone(),
                strides_cal: Arc::new(|_| vec![]),
                span: node.span,
            };
            nodes.insert(*id, srg_node);
        }
        let srg = Srg {
            nodes,
            tensors: ctx.nodes.clone(),
        };
        let schedule = srg.create_schedule(&order);
        println!("{}", schedule);
    }

    #[test]
    fn test_add_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let c = ctx.add(&a, &b);
        let order = [a.id, b.id, c.id];

        let mut nodes = HashMap::new();
        for (id, node) in ctx.nodes.borrow().iter() {
            let srg_node = SrgNode {
                id: *id,
                shape: node.shape.clone(),
                inputs: node.inputs.clone(),
                outputs: Arc::new(
                    ctx.nodes
                        .borrow()
                        .iter()
                        .filter_map(|(k, v)| {
                            if v.inputs.contains(id) { Some(*k) } else { None }
                        })
                        .collect()
                ),
                op: node.op.clone(),
                strides_cal: Arc::new(|_| vec![]),
                span: node.span,
            };
            nodes.insert(*id, srg_node);
        }
        let srg = Srg {
            nodes,
            tensors: ctx.nodes.clone(),
        };
        let schedule = srg.create_schedule(&order);
        println!("{}", schedule);
    }

    #[test]
    fn test_add_broadcast_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &1i64, &o], Dtype::F32);
        let b = ctx.placeholder(&[&m, &n, &1i64], Dtype::F32);
        let c = ctx.add(&a, &b);
        let order = [a.id, b.id, c.id];

        let mut nodes = HashMap::new();
        for (id, node) in ctx.nodes.borrow().iter() {
            let srg_node = SrgNode {
                id: *id,
                shape: node.shape.clone(),
                inputs: node.inputs.clone(),
                outputs: Arc::new(
                    ctx.nodes
                        .borrow()
                        .iter()
                        .filter_map(|(k, v)| {
                            if v.inputs.contains(id) { Some(*k) } else { None }
                        })
                        .collect()
                ),
                op: node.op.clone(),
                strides_cal: Arc::new(|_| vec![]),
                span: node.span,
            };
            nodes.insert(*id, srg_node);
        }
        let srg = Srg {
            nodes,
            tensors: ctx.nodes.clone(),
        };
        let schedule = srg.create_schedule(&order);
        println!("{}", schedule);
    }

    #[test]
    fn test_sum_broadcast_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.sum(&a, &0f32, &[1]);
        let order = [a.id, b.id];
        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        println!("{}", schedule);
    }

    #[test]
    fn test_sum_all_broadcast_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.sum(&a, &0f32, &[0]);
        let c = ctx.sum(&b, &0f32, &[0]);
        let e = ctx.sum(&c, &0f32, &[0]);
        let order = [a.id, b.id, c.id, e.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        println!("{}", schedule);
    }

    #[test]
    fn test_sum_all_broadcast_schedule2() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.sum(&a, &0f32, &[0, 1, 2]);
        let order = [a.id, b.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        println!("{}", schedule);
    }

    #[test]
    fn test_schedule3() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.placeholder(&[&1i64, &1i64, &1i64], Dtype::F32);
        let c = ctx.add(&a, &b);
        let sum = ctx.sum(&c, &0f32, &[2]);
        let reshaped = ctx.reshape(&sum, &[&m, &n, &1i64]);
        let add = ctx.add(&c, &reshaped);
        let sin = ctx.sin(&add);
        let sum2 = ctx.sum(&sin, &0f32, &[2]);
        let reshaped2 = ctx.reshape(&sum2, &[&m, &n, &1i64]);
        let add2 = ctx.add(&sin, &reshaped2);
        let order = [
            a.id,
            b.id,
            c.id,
            sum.id,
            reshaped.id,
            add.id,
            sin.id,
            sum2.id,
            reshaped2.id,
            add2.id,
        ];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);

        let func = schedule.to_function();
        println!("{}", func);
    }

    #[test]
    fn test_slice() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let a = ctx.placeholder(&[&m, &n], Dtype::F32);
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1));
        let b = ctx.slice(
            &a,
            &[
                (&0i64, &(&m.clone().into() - &one), &2i64),
                (&0i64, &(&n.clone().into() - &one), &2i64),
            ]
        );
        let c = ctx.slice(
            &a,
            &[
                (&1i64, &m, &2i64),
                (&1i64, &n, &2i64),
            ]
        );
        let add = ctx.add(&b, &c);
        let sum = ctx.sum(&add, &0f32, &[0]);
        let order = [a.id, b.id, c.id, add.id, sum.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        println!("{}", func);
    }
}
