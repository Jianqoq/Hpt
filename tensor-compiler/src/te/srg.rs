use std::{ collections::{ HashMap, HashSet }, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        tensor_load::{ Flag, TensorLoad },
        utils::var,
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
    pub(crate) fn create_strides_cal(&mut self, sorted: &[usize]) {
        for id in sorted {
            let inputs = self.nodes[&id].inputs.clone();
            if inputs.len() == 0 {
                let node = self.nodes.get_mut(&id).unwrap();
                let node_shape = node.shape.clone();
                let tensor_sc = self.tensors.borrow().get(&id).unwrap().strides_cal.clone();
                let input_func = Arc::new(move |map: &HashMap<Arc<String>, i64>| {
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
                let node = self.nodes.get_mut(&id).unwrap();
                let func = self.tensors.borrow().get(&id).unwrap().strides_cal.clone()(input_funcs);
                node.strides_cal = func;
            }
        }
    }

    pub fn create_schedule(&mut self, sorted: &[usize]) -> Schedule {
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
                                flag: Flag::default(),
                            },
                            false,
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
                    out_id: *id,
                    dtype: node.dtype.clone(),
                    begins: (0..node.shape.len()).map(|_| (0i64).into()).collect(),
                    steps: (0..node.shape.len()).map(|_| (1i64).into()).collect(),
                    axes: (0..node.shape.len()).map(|x| var(format!("ax{}", x)).into()).collect(),
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
        let nodes = self.tensors
            .borrow()
            .iter()
            .filter(|(_, x)| { sorted.contains(&x.id()) })
            .map(|(k, x)| (*k, x.clone()))
            .collect::<HashMap<usize, Tensor>>();
        Schedule { qa, nodes: RcMut::new(nodes), strides_cal }
    }
}
