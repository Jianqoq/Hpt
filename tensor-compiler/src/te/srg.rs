use std::{ collections::{ HashMap, VecDeque }, sync::Arc };

use tensor_common::{
    layout::Layout,
    shape::Shape,
    shape_utils::{ is_reshape_possible, predict_broadcast_shape, try_pad_shape },
    strides::Strides,
    strides_utils::{ preprocess_strides, shape_to_strides },
};
use tensor_types::dtype::Dtype;

use crate::halide::{
    exprs::{ Alloca, Call, Int, Let },
    let_stmt::LetStmt,
    prime_expr::PrimeExpr,
    seq_stmt::Seq,
    stmt::Stmt,
    variable::Variable,
};

use super::{ operation::Operation, srg_node::SrgNode };

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
                    vec![shape_to_strides(&real_shape).inner().clone()]
                });
                node.strides_cal = func;
            } else {
                match node.op.clone() {
                    Operation::Reshape(new_shape) => {
                        let prev_func = node.inputs[0].strides_cal.clone();
                        let prev_reduce_dim = node.inputs[0].reduced_dim;
                        let prev_shape = node.inputs[0].shape.clone();
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
                                    assert!(new.len() == new_shape.len());
                                    ret.push(new);
                                } else {
                                    panic!("Reshape not possible");
                                }
                            }
                            ret
                        });
                        node.reduced_dim = prev_reduce_dim;
                        node.strides_cal = func;
                    }
                    Operation::Transpose(axes) => {
                        let prev_func = node.inputs[0].strides_cal.clone();
                        let prev_reduce_dim = node.inputs[0].reduced_dim;
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
                        node.reduced_dim = prev_reduce_dim;
                        node.strides_cal = func;
                    }
                    Operation::Sum(axes) => {
                        let prev_func = node.inputs[0].strides_cal.clone();
                        let prev_reduce_dim = node.inputs[0].reduced_dim;
                        node.reduced_dim = prev_reduce_dim + axes.len();
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
                        node.strides_cal = func;
                    }
                    Operation::Sin => {
                        let prev_func = node.inputs[0].strides_cal.clone();
                        let prev_reduce_dim = node.inputs[0].reduced_dim;
                        let func = Arc::new(move |map: &HashMap<String, i64>| { prev_func(map) });
                        node.reduced_dim = prev_reduce_dim;
                        node.strides_cal = func;
                    }
                    Operation::Add => {
                        let lhs_prev_func = node.inputs[0].strides_cal.clone();
                        let rhs_prev_func = node.inputs[1].strides_cal.clone();
                        let lhs_reduce_dim = node.inputs[0].reduced_dim;
                        let rhs_reduce_dim = node.inputs[1].reduced_dim;
                        let lhs_shape = node.inputs[0].shape.clone();
                        let rhs_shape = node.inputs[1].shape.clone();
                        let res_shape = node.shape.clone();
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
                        node.strides_cal = func;
                    }
                    Operation::Slice(selections) => {
                        assert!(node.inputs.len() == 1);
                        assert!(node.inputs[0].inputs.len() == 0);
                    },
                    Operation::None => todo!(),
                }
            }
        }
    }
}