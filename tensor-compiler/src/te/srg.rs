use std::{ collections::{ HashMap, HashSet }, sync::Arc };

use tensor_common::{
    shape::Shape,
    shape_utils::{ is_reshape_possible, try_pad_shape },
    strides_utils::{ preprocess_strides, shape_to_strides },
};

use crate::{
    halide::{
        exprs::{ Call, Let, Load },
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        variable::Variable,
    },
    hlir::tensor_slice::TensorLoad,
    iter_var::IterVar,
    te::{
        hstrides::HStrides,
        idx_evaluator::IdxEvaluator,
        shape_utils::detect_broadcast_axes_expr,
        stages::{ Body, Stage },
    },
};

use super::{ operation::Operation, schedule::Schedule, srg_node::SrgNode };

pub struct Srg {
    pub(crate) nodes: HashMap<usize, SrgNode>,
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
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let _ = selections
                                .iter()
                                .map(|(start, end, step)| (
                                    IdxEvaluator::new(map).eval(start),
                                    IdxEvaluator::new(map).eval(end),
                                    IdxEvaluator::new(map).eval(step),
                                ))
                                .collect::<Vec<(i64, i64, i64)>>();
                            strides_cal(map)
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
        let mut stages = HashMap::new();
        for id in sorted {
            let node = &self.nodes[id];
            if node.inputs.len() == 0 {
                let body = Body::PrimeExpr(
                    (TensorLoad {
                        var: Variable::make(&format!("%{}", node.id)).into(),
                        begins: (0..node.shape.len())
                            .map(|_| (0i64).into())
                            .collect::<Vec<PrimeExpr>>()
                            .into(),
                        axes: (0..node.shape.len())
                            .map(|x| Variable::make(&format!("ax{}", x)).into())
                            .collect::<Vec<PrimeExpr>>()
                            .into(),
                        steps: (0..node.shape.len())
                            .map(|_| (1i64).into())
                            .collect::<Vec<PrimeExpr>>()
                            .into(),
                        strides: (0..node.shape.len())
                            .map(|idx|
                                Load::make(Variable::make(&format!("%{}.s", id)), idx).into()
                            )
                            .collect::<Vec<PrimeExpr>>()
                            .into(),
                    }).into()
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
                stages.insert(*id, stage);
                declared_vars.insert(format!("%{}_val", node.id));
            } else {
                match &node.op {
                    Operation::Reshape(reshape) => {
                        let input = stages[&node.inputs[0]].bodys.clone();
                        let stage = Stage {
                            dims: (0..reshape.len())
                                .map(|x|
                                    IterVar::new(0i64, reshape[x].clone(), 1i64, &format!("ax{}", x))
                                )
                                .collect(),
                            bodys: input,
                            id: *id,
                        };
                        stages.insert(*id, stage);
                    }
                    Operation::Transpose(_) => {}
                    Operation::Sum(axes, init) => {
                        let input = stages[&node.inputs[0]].bodys.clone();
                        let stage = Stage {
                            dims: axes
                                .iter()
                                .enumerate()
                                .map(|(idx, x)| {
                                    let mut iter_var = stages[&node.inputs[0]].dims[*x].clone();
                                    iter_var.set_var(
                                        Variable::new(format!("red{}_ax{}", node.id, idx))
                                    );
                                    iter_var
                                })
                                .collect(),
                            bodys: input,
                            id: *id,
                        };
                        let dims = stages[&node.inputs[0]].dims
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| !axes.contains(idx))
                            .enumerate()
                            .map(|(idx, (_, x))| {
                                let mut iter_var = x.clone();
                                iter_var.set_var(Variable::new(format!("ax{}", idx)));
                                iter_var
                            })
                            .collect::<Vec<_>>();
                        let new_body = vec![
                            Body::Stmt(
                                LetStmt::make(
                                    &Variable::make(&format!("%{}_val", id)),
                                    init,
                                    Stmt::None
                                ).into()
                            ),
                            Body::Stage(*id)
                        ];
                        let new_stage = Stage {
                            dims,
                            bodys: new_body,
                            id: node.inputs[0],
                        };
                        stages.insert(*id, stage);
                        stages.insert(node.inputs[0], new_stage);
                        declared_vars.insert(format!("%{}_val", id));
                    }
                    Operation::Sin => {
                        let mut input = stages[&node.inputs[0]].bodys.clone();
                        input.push(
                            Body::Stmt(
                                LetStmt::make(
                                    &Variable::make(&format!("%{}_val", id)),
                                    Call::make(
                                        "sin",
                                        &[Variable::new(format!("%{}_val", node.inputs[0]))]
                                    ),
                                    Stmt::None
                                ).into()
                            )
                        );
                        let stage = Stage {
                            dims: stages[&node.inputs[0]].dims.clone(),
                            bodys: input,
                            id: *id,
                        };
                        stages.insert(*id, stage);
                    }
                    Operation::Add => {
                        let a_shape = &self.nodes[&node.inputs[0]].shape;
                        let b_shape = &self.nodes[&node.inputs[1]].shape;
                        let _ = detect_broadcast_axes_expr(a_shape, b_shape);
                        let dims = (0..node.shape.len())
                            .map(|x|
                                IterVar::new(0i64, node.shape[x].clone(), 1i64, &format!("ax{}", x))
                            )
                            .collect::<Vec<_>>();
                        let mut a_body = stages[&node.inputs[0]].bodys.clone();
                        let b_body = stages[&node.inputs[1]].bodys.clone();
                        a_body.extend(b_body);
                        a_body.push(
                            Body::Stmt(
                                LetStmt::make(
                                    &Variable::make(&format!("%{}_val", id)),
                                    Variable::new(format!("%{}_val", node.inputs[0])) +
                                        Variable::new(format!("%{}_val", node.inputs[1])),
                                    Stmt::None
                                ).into()
                            )
                        );
                        let stage = Stage {
                            dims,
                            bodys: a_body,
                            id: *id,
                        };
                        stages.insert(*id, stage);
                    }
                    Operation::Slice(_) => {}
                    Operation::None => {}
                }
            }
        }
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::{ collections::HashMap, sync::Arc };

    use tensor_types::dtype::Dtype;

    use crate::te::{ context::Context, srg_node::SrgNode };

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
                op: node.op.clone(),
                strides_cal: Arc::new(|_| vec![]),
                span: node.span,
            };
            nodes.insert(*id, srg_node);
        }
        let mut srg = Srg {
            nodes,
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
}
