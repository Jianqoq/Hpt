use std::{ collections::{ HashMap, VecDeque }, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;
use tensor_types::dtype::Dtype;

use crate::halide::{
    exprs::{ Alloca, Call, Int, Let },
    let_stmt::LetStmt,
    prime_expr::PrimeExpr,
    seq_stmt::Seq,
    stmt::Stmt,
    variable::Variable,
};

use super::srg_node::SrgNode;

pub struct Srg {
    pub(crate) nodes: HashMap<usize, SrgNode>,
}

impl Srg {
    pub fn fuse(&mut self, sorted: VecDeque<usize>) {
        for id in sorted {
            let node = self.nodes.get_mut(&id).unwrap();
            if node.inputs.len() == 0 {
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
                    shape_to_strides(&real_shape).inner().clone()
                });
                node.strides_cal = func;
                node.using.push(id);
            } else {
                if node.inputs.len() == 1 {
                } else if node.inputs.len() == 2 {
                } else {
                }
            }
        }
    }
}
