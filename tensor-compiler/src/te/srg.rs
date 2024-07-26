use std::{ collections::{ HashMap, HashSet }, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        tensor_load::TensorLoad,
        variable::Variable,
    },
    iter_var::IterVar,
    te::{ hstrides::HStrides, idx_evaluator::IdxEvaluator, stages::{ Body, Stage } },
};

use super::{ rc_mut::RcMut, schedule::Schedule, srg_node::SrgNode, tensor::Tensor };

pub struct Srg {
    pub(crate) nodes: HashMap<usize, SrgNode>,
    pub(crate) tensors: RcMut<HashMap<usize, Tensor>>,
}

impl Srg {
    pub fn create_strides_cal(&mut self, sorted: &[usize]) {
        for id in sorted {
            let inputs = self.nodes[&id].inputs.clone();
            if inputs.len() == 0 {
                let node = self.nodes.get_mut(&id).unwrap();
                let node_shape = node.shape.clone();
                let tensor_sc = self.tensors.borrow().get(&id).unwrap().strides_cal.clone();
                let input_func = Arc::new(move |map: &HashMap<String, i64>| {
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
                let func = tensor_sc(vec![input_func]);
                node.strides_cal = func;
            } else {
                let inputs = self.nodes[&id].inputs.clone();
                let mut input_funcs = vec![];
                for i in inputs.iter() {
                    input_funcs.push(self.nodes[&i].strides_cal.clone());
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
                let mut inputs = vec![];
                for i in node.inputs.iter() {
                    inputs.push(qa.get(i).unwrap().0.clone());
                }
                let res = (self.tensors.borrow().get(&node.id).unwrap().body_gen)(
                    inputs,
                    node.is_output(),
                    node.id
                );
                qa.insert(*id, (res, node.is_output()));
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
        to_prim_expr::ToPrimeExpr,
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
    fn test_add_broadcast_diff_len_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&o, &m, &1i64], Dtype::F32);
        let b = ctx.placeholder(&[&m, &n], Dtype::F32);
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

    #[test]
    fn test_pad() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let a = ctx.placeholder(&[&m, &n], Dtype::F32);
        let b = ctx.pad(
            &a,
            &[
                (&5i64, &5i64),
                (&5i64, &5i64),
            ],
            &1f32
        );
        let c = ctx.placeholder(
            &[&(&m.into() + &(10i64).to_prime_expr()), &(&n.into() + &(10i64).to_prime_expr())],
            Dtype::F32
        );
        let d = ctx.sin(&b);
        let e = ctx.add(&d, &c);
        let order = [a.id, b.id, c.id, d.id, e.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        println!("{}", func);
    }
}
