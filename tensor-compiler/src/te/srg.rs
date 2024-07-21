use std::{ collections::{ HashMap, VecDeque }, sync::Arc };

use tensor_common::{
    shape::Shape,
    shape_utils::{ is_reshape_possible, try_pad_shape },
    strides_utils::{ preprocess_strides, shape_to_strides },
};

use crate::halide::prime_expr::PrimeExpr;

use super::{ operation::Operation, srg_node::SrgNode };

pub struct Srg {
    pub(crate) nodes: HashMap<usize, SrgNode>,
}

impl Srg {
    pub fn create_strides_cal(&mut self, sorted: &VecDeque<usize>) {
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
                        .map(|x| {
                            match x {
                                PrimeExpr::Variable(var) => map[var.name()],
                                _ => x.evaluate_i64(),
                            }
                        })
                        .collect::<Vec<i64>>();
                    vec![shape_to_strides(&real_shape).inner().clone()]
                });
                node.strides_cal = func;
            } else {
                match node_op {
                    Operation::Reshape(new_shape) => {
                        let (prev_func, prev_reduce_dim, prev_shape) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].strides_cal.clone(),
                                self.nodes[&node.inputs[0]].reduced_dim,
                                self.nodes[&node.inputs[0]].shape.clone(),
                            )
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let prev_strides = prev_func(map);
                            let new_shape = new_shape
                                .iter()
                                .map(|x| {
                                    match x {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => x.evaluate_i64(),
                                    }
                                })
                                .collect::<Vec<i64>>();
                            let prev_shape = prev_shape
                                .iter()
                                .map(|x| {
                                    match x {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => x.evaluate_i64(),
                                    }
                                })
                                .collect::<Vec<i64>>();
                            let new_shape = Shape::from(new_shape);
                            let prev_shape = Shape::from(prev_shape);
                            let mut ret = vec![];
                            for i in prev_strides.into_iter() {
                                let masked_strides = &i[..i.len() - prev_reduce_dim];
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
                                    for i in i[i.len() - prev_reduce_dim..].iter() {
                                        new.push(*i);
                                    }
                                    assert!(new.len() == new_shape.len() + prev_reduce_dim);
                                    ret.push(new);
                                } else {
                                    panic!("Reshape not possible, {}", span_location);
                                }
                            }
                            ret
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.reduced_dim = prev_reduce_dim;
                        node.strides_cal = func;
                    }
                    Operation::Transpose(axes) => {
                        let (prev_func, prev_reduce_dim) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].strides_cal.clone(),
                                self.nodes[&node.inputs[0]].reduced_dim,
                            )
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let prev_strides = prev_func(map);
                            let mut ret = vec![];
                            for strides in prev_strides.into_iter() {
                                let masked = &strides[..strides.len() - prev_reduce_dim];
                                let mut new_strides = vec![];
                                for i in axes.iter() {
                                    new_strides.push(masked[*i]);
                                }
                                for i in strides[strides.len() - prev_reduce_dim..].iter() {
                                    new_strides.push(*i);
                                }
                                assert!(new_strides.len() == strides.len());
                                ret.push(new_strides);
                            }
                            ret
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.reduced_dim = prev_reduce_dim;
                        node.strides_cal = func;
                    }
                    Operation::Sum(axes) => {
                        let (prev_func, prev_reduce_dim) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].strides_cal.clone(),
                                self.nodes[&node.inputs[0]].reduced_dim,
                            )
                        };
                        let axes_len = axes.len();
                        let func = Arc::new(move |map: &HashMap<String, i64>| {
                            let prev_strides = prev_func(map);
                            let mut ret = vec![];
                            for strides in prev_strides.into_iter() {
                                let masked = &strides[..strides.len() - prev_reduce_dim];
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
                                for i in strides[strides.len() - prev_reduce_dim..].iter() {
                                    new_strides.push(*i);
                                }
                                assert!(new_strides.len() == strides.len());
                                ret.push(new_strides);
                            }
                            ret
                        });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.reduced_dim = prev_reduce_dim + axes_len;
                        node.strides_cal = func;
                    }
                    Operation::Sin => {
                        let (prev_func, prev_reduce_dim) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].strides_cal.clone(),
                                self.nodes[&node.inputs[0]].reduced_dim,
                            )
                        };
                        let func = Arc::new(move |map: &HashMap<String, i64>| { prev_func(map) });
                        let node = self.nodes.get_mut(&id).unwrap();
                        node.reduced_dim = prev_reduce_dim;
                        node.strides_cal = func;
                    }
                    Operation::Add => {
                        let (lhs_prev_func, lhs_reduce_dim) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[0]].strides_cal.clone(),
                                self.nodes[&node.inputs[0]].reduced_dim,
                            )
                        };
                        let (rhs_prev_func, rhs_reduce_dim) = {
                            let node = &self.nodes[&id];
                            (
                                self.nodes[&node.inputs[1]].strides_cal.clone(),
                                self.nodes[&node.inputs[1]].reduced_dim,
                            )
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
                                .map(|x| {
                                    match x {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => x.evaluate_i64(),
                                    }
                                })
                                .collect::<Vec<i64>>();

                            let rhs_real_shape = rhs_shape
                                .iter()
                                .map(|x| {
                                    match x {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => x.evaluate_i64(),
                                    }
                                })
                                .collect::<Vec<i64>>();

                            let res_real_shape = res_shape
                                .iter()
                                .map(|x| {
                                    match x {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => x.evaluate_i64(),
                                    }
                                })
                                .collect::<Vec<i64>>();

                            let mut lhs_strides_vec = vec![];
                            for strides in lhs_strides.into_iter() {
                                let masked = &strides[..strides.len() - lhs_reduce_dim];
                                assert!(masked.len() == lhs_real_shape.len());
                                let padded = try_pad_shape(&lhs_real_shape, res_real_shape.len());
                                let new = preprocess_strides::<_, _, i64>(&padded, masked);
                                let mut new_strides = vec![];
                                for i in new.iter() {
                                    new_strides.push(*i);
                                }
                                for i in strides[strides.len() - lhs_reduce_dim..].iter() {
                                    new_strides.push(*i);
                                }
                                lhs_strides_vec.push(new_strides);
                            }
                            let mut rhs_strides_vec = vec![];
                            for strides in rhs_strides.into_iter() {
                                let masked = &strides[..strides.len() - rhs_reduce_dim];
                                assert!(masked.len() == rhs_real_shape.len());
                                let padded = try_pad_shape(&rhs_real_shape, res_real_shape.len());
                                let new = preprocess_strides::<_, _, i64>(&padded, masked);
                                let mut new_strides = vec![];
                                for i in new.iter() {
                                    new_strides.push(*i);
                                }
                                for i in strides[strides.len() - rhs_reduce_dim..].iter() {
                                    new_strides.push(*i);
                                }
                                rhs_strides_vec.push(new_strides);
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
                                    match start {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => start.evaluate_i64(),
                                    },
                                    match end {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => end.evaluate_i64(),
                                    },
                                    match step {
                                        PrimeExpr::Variable(var) => map[var.name()],
                                        _ => step.evaluate_i64(),
                                    },
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
}

#[cfg(test)]
mod tests {
    use std::{ collections::{ HashMap, VecDeque }, sync::Arc };

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
        let order = VecDeque::from(
            vec![a.id, b.id, c.id, d.id, e.id, f.id, g.id, h.id, i.id, j.id]
        );

        let mut nodes = HashMap::new();
        for (id, node) in ctx.nodes.borrow().iter() {
            let srg_node = SrgNode {
                id: *id,
                shape: node.shape.clone(),
                inputs: node.inputs.clone(),
                op: node.op.clone(),
                reduced_dim: 0,
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

        for idx in order.iter() {
            let node = &srg.nodes[idx];
            if node.id == 8 {
                println!();
            }
            let strides = (node.strides_cal)(&var_map);
            // println!("======{:?}=======", strides);
        }
    }
}
