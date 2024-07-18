use std::{ collections::VecDeque, sync::Arc };

use std::collections::{ HashMap, HashSet };

use crate::halide::assign_stmt::AssignStmt;
use crate::halide::exprs::Gt;
use crate::halide::if_stmt::IfThenElse;
use crate::halide::inplace_store_stmt::{ InplaceAdd, InplaceMul };
use crate::halide::let_stmt::LetStmt;
use crate::halide::prime_expr::PrimeExpr;
use crate::{
    edges::Edges,
    halide::{
        exprs::Load,
        for_stmt::For,
        seq_stmt::Seq,
        stmt::Stmt,
        store_stmt::StoreStmt,
        substitute::subsititue_expr::SubstituteExpr,
        traits::{ AccepterMutate, MutatorGetSet },
        variable::Variable,
    },
    hlir::schedule::schedule::{ Node, RcMut, Stage },
    iter_var::IterVar,
    to_prim_expr::ToPrimeExpr,
};

pub fn build_nested_for<T: Into<Stmt>>(iter_vars: &[IterVar], main_stmt: T) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(idx: usize, iter_vars: &[IterVar], main_stmt: T) -> Stmt {
        if idx == iter_vars.len() {
            main_stmt.into()
        } else {
            let to_add = For::make(
                iter_vars[idx].var(),
                iter_vars[idx].start(),
                iter_vars[idx].end(),
                iter_vars[idx].step(),
                build_recursive(idx + 1, iter_vars, main_stmt)
            );
            Stmt::For(to_add)
        }
    }

    build_recursive(0, iter_vars, main_stmt)
}

fn topo(nodes: &HashMap<usize, Arc<String>>, edges: Edges<usize>) -> Option<VecDeque<usize>> {
    let mut in_degree = HashMap::new();
    let mut queue = VecDeque::new();
    let mut order = VecDeque::new();
    let edges = edges.invert();
    // calculate in degree
    for (&node_id, _) in nodes.iter() {
        in_degree.entry(node_id).or_insert(0);
        let edges = edges.get(&node_id);
        if let Some(edges) = edges {
            for &target in edges {
                *in_degree.entry(target).or_insert(0) += 1;
            }
        }
    }

    // push nodes with in degree 0 to queue
    for (&node_id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node_id);
        }
    }

    // topological sort
    while let Some(node_id) = queue.pop_front() {
        order.push_back(node_id);
        if let Some(_) = nodes.get(&node_id) {
            let edges = edges.get(&node_id);
            if let Some(edges) = edges {
                for &target in edges {
                    let degree = in_degree.get_mut(&target).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(target);
                    }
                }
            }
        }
    }

    // check if there is a cycle
    if order.len() == nodes.len() {
        Some(order)
    } else {
        None // cycle detected
    }
}

pub fn build_nested_for_helper(stage: RcMut<Stage>, iter_vars: &[RcMut<Node>]) -> Stmt {
    fn build_recursive(idx: usize, stage: RcMut<Stage>, iter_vars: &[RcMut<Node>]) -> Stmt {
        let mut seq = None;
        let store = if iter_vars.len() == 0 || idx == iter_vars.len() - 1 {
            let mut subs_expr = SubstituteExpr::new();
            let mut store_indices_root = vec![];
            for origin in stage.borrow().root.borrow().iter() {
                if
                    stage
                        .borrow()
                        .freezed_leaf.borrow()
                        .iter()
                        .any(|x| x.as_ptr() == origin.as_ptr())
                {
                    continue;
                }
                // println!("{} -> {}", origin.borrow().var(), origin.borrow().expr());
                subs_expr.add_replacement(
                    origin.borrow().var().to_prime_expr(),
                    origin.borrow().expr().clone()
                );
                store_indices_root.push(origin.borrow().expr().clone());
            }
            let mut store_indices_freezed = vec![];
            for (origin, target) in stage
                .borrow()
                .freezed_leaf.borrow()
                .iter()
                .zip(stage.borrow().freezed_target.borrow().iter()) {
                // println!("{} -> {}", origin.borrow().var().to_prime_expr(), target.borrow().expr());
                subs_expr.add_replacement(
                    origin.borrow().var().to_prime_expr(),
                    target.borrow().expr()
                );
                if
                    stage
                        .borrow()
                        .root.borrow()
                        .iter()
                        .any(|x| x.as_ptr() == origin.as_ptr())
                {
                    store_indices_freezed.push(target.borrow().expr().clone());
                }
            }
            stage.borrow().body.accept_mutate(&mut subs_expr);

            let load_strides = (0..stage.borrow().root.borrow().len()).map(|x| {
                Load::make(format!("{}.strides", stage.borrow().name.as_ref()), x)
            });
            store_indices_freezed.extend(store_indices_root.iter().cloned());
            match subs_expr.expr() {
                PrimeExpr::Reduce(reduce) => {
                    match reduce.op() {
                        "sum" => {
                            let store_init = StoreStmt::make(
                                &Variable::make(&stage.borrow().name),
                                store_indices_freezed
                                    .iter()
                                    .zip(load_strides.clone())
                                    .map(|(x, strides)| x.clone() * strides.into())
                                    .reduce(|x, y| x + y)
                                    .unwrap(),
                                &reduce.identity()[0]
                            );
                            let iter_var = &reduce.iter_vars()[0];
                            let inplace_add = InplaceAdd::make(
                                Load::make(
                                    Variable::make(&stage.borrow().name),
                                    store_indices_freezed
                                        .iter()
                                        .zip(load_strides.clone())
                                        .map(|(x, strides)| x.clone() * strides.into())
                                        .reduce(|x, y| x + y)
                                        .unwrap()
                                ),
                                &reduce.expr()[0]
                            );
                            let for_stmt = For::make(
                                iter_var.var(),
                                iter_var.start(),
                                iter_var.end(),
                                iter_var.step(),
                                Stmt::InplaceAdd(inplace_add)
                            );
                            Some(
                                Stmt::Seq(
                                    Seq::make(
                                        vec![Stmt::StoreStmt(store_init), Stmt::For(for_stmt)]
                                    )
                                )
                            )
                        }
                        "prod" => {
                            let store_init = StoreStmt::make(
                                &Variable::make(&stage.borrow().name),
                                store_indices_freezed
                                    .iter()
                                    .zip(load_strides.clone())
                                    .map(|(x, strides)| x.clone() * strides.into())
                                    .reduce(|x, y| x + y)
                                    .unwrap(),
                                &reduce.identity()[0]
                            );
                            let iter_var = &reduce.iter_vars()[0];
                            let inplace_mul = InplaceMul::make(
                                Load::make(
                                    Variable::make(&stage.borrow().name),
                                    store_indices_freezed
                                        .iter()
                                        .zip(load_strides.clone())
                                        .map(|(x, strides)| x.clone() * strides.into())
                                        .reduce(|x, y| x + y)
                                        .unwrap()
                                ),
                                &reduce.expr()[0]
                            );
                            let for_stmt = For::make(
                                iter_var.var(),
                                iter_var.start(),
                                iter_var.end(),
                                iter_var.step(),
                                Stmt::InplaceMul(inplace_mul)
                            );
                            Some(
                                Stmt::Seq(
                                    Seq::make(
                                        vec![Stmt::StoreStmt(store_init), Stmt::For(for_stmt)]
                                    )
                                )
                            )
                        }
                        "argmax" => {
                            let store_init = StoreStmt::make(
                                &Variable::make(&stage.borrow().name),
                                store_indices_freezed
                                    .iter()
                                    .zip(load_strides.clone())
                                    .map(|(x, strides)| x.clone() * strides.into())
                                    .reduce(|x, y| x + y)
                                    .unwrap(),
                                &reduce.identity()[0]
                            );
                            let mut init_neg_inf = LetStmt::make(
                                &Variable::new(format!("{}_tmp", &stage.borrow().name)),
                                &reduce.identity()[1],
                                Stmt::None
                            );
                            let iter_var = &reduce.iter_vars()[0];
                            let cond = Gt::make(
                                &reduce.expr()[0],
                                Variable::new(format!("{}_tmp", &stage.borrow().name))
                            );
                            let true_stmt = Stmt::Seq(
                                Seq::make(
                                    vec![
                                        Stmt::AssignStmt(
                                            AssignStmt::make(
                                                &Variable::new(
                                                    format!("{}_tmp", &stage.borrow().name)
                                                ),
                                                &reduce.expr()[0]
                                            )
                                        ),
                                        Stmt::StoreStmt(
                                            StoreStmt::make(
                                                &Variable::make(&stage.borrow().name),
                                                store_indices_freezed
                                                    .iter()
                                                    .zip(load_strides.clone())
                                                    .map(|(x, strides)| x.clone() * strides.into())
                                                    .reduce(|x, y| x + y)
                                                    .unwrap(),
                                                iter_var.var()
                                            )
                                        )
                                    ]
                                )
                            );
                            let if_stmt = IfThenElse::make(cond, true_stmt, Stmt::None);
                            let for_stmt = For::make(
                                iter_var.var(),
                                iter_var.start(),
                                iter_var.end(),
                                iter_var.step(),
                                Stmt::IfThenElse(if_stmt)
                            );
                            init_neg_inf.set_body(
                                Stmt::Seq(
                                    Seq::make(vec![Stmt::StoreStmt(store_init), for_stmt.into()])
                                )
                            );
                            Some(Stmt::LetStmt(init_neg_inf))
                        }
                        "argmin" => {
                            let store_init = StoreStmt::make(
                                &Variable::make(&stage.borrow().name),
                                store_indices_freezed
                                    .iter()
                                    .zip(load_strides.clone())
                                    .map(|(x, strides)| x.clone() * strides.into())
                                    .reduce(|x, y| x + y)
                                    .unwrap(),
                                &reduce.identity()[0]
                            );
                            let mut init_inf = LetStmt::make(
                                &Variable::new(format!("{}_tmp", &stage.borrow().name)),
                                &reduce.identity()[1],
                                Stmt::None
                            );
                            let iter_var = &reduce.iter_vars()[0];
                            let cond = Gt::make(
                                &reduce.expr()[0],
                                Variable::new(format!("{}_tmp", &stage.borrow().name))
                            );
                            let true_stmt = Stmt::Seq(
                                Seq::make(
                                    vec![
                                        Stmt::AssignStmt(
                                            AssignStmt::make(
                                                &Variable::new(
                                                    format!("{}_tmp", &stage.borrow().name)
                                                ),
                                                &reduce.expr()[0]
                                            )
                                        ),
                                        Stmt::StoreStmt(
                                            StoreStmt::make(
                                                &Variable::make(&stage.borrow().name),
                                                store_indices_freezed
                                                    .iter()
                                                    .zip(load_strides.clone())
                                                    .map(|(x, strides)| x.clone() * strides.into())
                                                    .reduce(|x, y| x + y)
                                                    .unwrap(),
                                                iter_var.var()
                                            )
                                        )
                                    ]
                                )
                            );
                            let if_stmt = IfThenElse::make(cond, true_stmt, Stmt::None);
                            let for_stmt = For::make(
                                iter_var.var(),
                                iter_var.start(),
                                iter_var.end(),
                                iter_var.step(),
                                Stmt::IfThenElse(if_stmt)
                            );
                            init_inf.set_body(
                                Stmt::Seq(
                                    Seq::make(vec![Stmt::StoreStmt(store_init), for_stmt.into()])
                                )
                            );
                            Some(Stmt::LetStmt(init_inf))
                        }
                        "max" => {
                            let store_init = StoreStmt::make(
                                &Variable::make(&stage.borrow().name),
                                store_indices_freezed
                                    .iter()
                                    .zip(load_strides.clone())
                                    .map(|(x, strides)| x.clone() * strides.into())
                                    .reduce(|x, y| x + y)
                                    .unwrap(),
                                &reduce.identity()[0]
                            );
                            let iter_var = &reduce.iter_vars()[0];
                            let cond = Gt::make(
                                &reduce.expr()[0],
                                Load::make(
                                    Variable::make(&stage.borrow().name),
                                    store_indices_freezed
                                        .iter()
                                        .zip(load_strides.clone())
                                        .map(|(x, strides)| x.clone() * strides.into())
                                        .reduce(|x, y| x + y)
                                        .unwrap()
                                )
                            );
                            let true_stmt = Stmt::StoreStmt(
                                StoreStmt::make(
                                    &Variable::make(&stage.borrow().name),
                                    store_indices_freezed
                                        .iter()
                                        .zip(load_strides.clone())
                                        .map(|(x, strides)| x.clone() * strides.into())
                                        .reduce(|x, y| x + y)
                                        .unwrap(),
                                    &reduce.expr()[0]
                                )
                            );
                            let if_stmt = IfThenElse::make(cond, true_stmt, Stmt::None);
                            let for_stmt = For::make(
                                iter_var.var(),
                                iter_var.start(),
                                iter_var.end(),
                                iter_var.step(),
                                Stmt::IfThenElse(if_stmt)
                            );
                            Some(
                                Stmt::Seq(
                                    Seq::make(
                                        vec![Stmt::StoreStmt(store_init), Stmt::For(for_stmt)]
                                    )
                                )
                            )
                        }
                        "min" => {
                            let store_init = StoreStmt::make(
                                &Variable::make(&stage.borrow().name),
                                store_indices_freezed
                                    .iter()
                                    .zip(load_strides.clone())
                                    .map(|(x, strides)| x.clone() * strides.into())
                                    .reduce(|x, y| x + y)
                                    .unwrap(),
                                &reduce.identity()[0]
                            );
                            let iter_var = &reduce.iter_vars()[0];
                            let cond = Gt::make(
                                Load::make(
                                    Variable::make(&stage.borrow().name),
                                    store_indices_freezed
                                        .iter()
                                        .zip(load_strides.clone())
                                        .map(|(x, strides)| x.clone() * strides.into())
                                        .reduce(|x, y| x + y)
                                        .unwrap()
                                ),
                                &reduce.expr()[0]
                            );
                            let true_stmt = Stmt::StoreStmt(
                                StoreStmt::make(
                                    &Variable::make(&stage.borrow().name),
                                    store_indices_freezed
                                        .iter()
                                        .zip(load_strides.clone())
                                        .map(|(x, strides)| x.clone() * strides.into())
                                        .reduce(|x, y| x + y)
                                        .unwrap(),
                                    &reduce.expr()[0]
                                )
                            );
                            let if_stmt = IfThenElse::make(cond, true_stmt, Stmt::None);
                            let for_stmt = For::make(
                                iter_var.var(),
                                iter_var.start(),
                                iter_var.end(),
                                iter_var.step(),
                                Stmt::IfThenElse(if_stmt)
                            );
                            Some(
                                Stmt::Seq(
                                    Seq::make(
                                        vec![Stmt::StoreStmt(store_init), Stmt::For(for_stmt)]
                                    )
                                )
                            )
                        }
                        _ => { panic!("Not implemented") }
                    }
                }
                _ => {
                    let store = StoreStmt::make(
                        &Variable::make(&stage.borrow().name),
                        store_indices_freezed
                            .iter()
                            .zip(load_strides)
                            .map(|(x, strides)| x.clone() * strides.into())
                            .reduce(|x, y| x + y)
                            .unwrap(),
                        subs_expr.expr()
                    );
                    Some(Stmt::StoreStmt(store))
                }
            }
        } else {
            None
        };
        if iter_vars.len() > 0 {
            if
                let Some(stages) = stage
                    .borrow()
                    .attached_stage.borrow()
                    .get(&(iter_vars[idx].as_ptr() as usize))
            {
                let mut cnt = 0;
                let mut nodes_id = HashMap::new();
                let mut id_nodes = HashMap::new();

                let mut stages = stages
                    .iter()
                    .map(|x| x.clone())
                    .collect::<Vec<_>>();

                if idx == iter_vars.len() - 1 {
                    stages.push(stage.clone());
                }

                for stage in stages.iter() {
                    if let None = nodes_id.get(&stage.borrow().name) {
                        nodes_id.insert(stage.borrow().name.clone(), cnt);
                        id_nodes.insert(cnt, stage.borrow().name.clone());
                        cnt += 1;
                    }
                    for input in stage.borrow().inputs.iter() {
                        if let None = nodes_id.get(input) {
                            nodes_id.insert(input.clone(), cnt);
                            id_nodes.insert(cnt, input.clone());
                            cnt += 1;
                        }
                    }
                }
                let mut edges = Edges::new();

                for stage in stages.iter() {
                    let node_id = nodes_id[&stage.borrow().name];
                    for input in stage.borrow().inputs.iter() {
                        let input_id = nodes_id[input];
                        edges.entry(node_id).or_insert(HashSet::new()).insert(input_id);
                    }
                }

                let sorted = topo(&id_nodes, edges).expect("Cycle detected");
                let mut vec = Vec::with_capacity(stages.len());

                if idx == iter_vars.len() - 1 {
                    for id in sorted.iter() {
                        let name = &id_nodes[id];
                        for i in stages.iter() {
                            if i.borrow().name == *name {
                                if i.borrow().name == stage.borrow().name {
                                    vec.push(store.clone().unwrap());
                                } else {
                                    let halide = i.borrow().to_halid();
                                    vec.push(halide);
                                }
                            }
                        }
                    }
                } else {
                    for id in sorted.iter() {
                        let name = &id_nodes[id];
                        for i in stages.iter() {
                            if i.borrow().name == *name {
                                let halide = i.borrow().to_halid();
                                vec.push(halide);
                            }
                        }
                    }
                }
                seq = Some(vec);
            }
        }

        if let Some(mut seq) = seq {
            let end: crate::halide::prime_expr::PrimeExpr = iter_vars[idx].borrow().end().clone();
            let shape = stage
                .borrow()
                .root.borrow()
                .iter()
                .map(|x| x.borrow().end().clone())
                .collect::<Vec<_>>();
            let mut subs_expr = SubstituteExpr::new();
            for (idx, x) in shape.iter().enumerate() {
                subs_expr.add_replacement(
                    x.clone(),
                    Load::make(Variable::make(&format!("{}.shape", stage.borrow().name)), idx)
                );
            }
            end.accept_mutate(&mut subs_expr);
            if idx == iter_vars.len() - 1 {
                Stmt::For(
                    For::make(
                        iter_vars[idx].borrow().var(),
                        iter_vars[idx].borrow().start(),
                        subs_expr.expr(),
                        iter_vars[idx].borrow().step(),
                        Seq::make(seq)
                    )
                )
            } else {
                seq.push(build_recursive(idx + 1, stage, iter_vars));
                Stmt::For(
                    For::make(
                        iter_vars[idx].borrow().var(),
                        iter_vars[idx].borrow().start(),
                        subs_expr.expr(),
                        iter_vars[idx].borrow().step(),
                        Seq::make(seq)
                    )
                )
            }
        } else {
            if iter_vars.len() > 0 && idx == iter_vars.len() - 1 {
                let end: crate::halide::prime_expr::PrimeExpr = iter_vars[idx]
                    .borrow()
                    .end()
                    .clone();
                let shape = stage
                    .borrow()
                    .root.borrow()
                    .iter()
                    .map(|x| x.borrow().end().clone())
                    .collect::<Vec<_>>();
                let mut subs_expr = SubstituteExpr::new();
                for (idx, x) in shape.iter().enumerate() {
                    subs_expr.add_replacement(
                        x.clone(),
                        Load::make(Variable::make(&format!("{}.shape", stage.borrow().name)), idx)
                    );
                }
                end.accept_mutate(&mut subs_expr);
                Stmt::For(
                    For::make(
                        iter_vars[idx].borrow().var(),
                        iter_vars[idx].borrow().start(),
                        subs_expr.expr(),
                        iter_vars[idx].borrow().step(),
                        store.unwrap()
                    )
                )
            } else {
                if iter_vars.len() == 0 {
                    store.unwrap()
                } else {
                    let end: crate::halide::prime_expr::PrimeExpr = iter_vars[idx]
                        .borrow()
                        .end()
                        .clone();
                    let shape = stage
                        .borrow()
                        .root.borrow()
                        .iter()
                        .map(|x| x.borrow().end().clone())
                        .collect::<Vec<_>>();
                    let mut subs_expr = SubstituteExpr::new();
                    for (idx, x) in shape.iter().enumerate() {
                        subs_expr.add_replacement(
                            x.clone(),
                            Load::make(
                                Variable::make(&format!("{}.shape", stage.borrow().name)),
                                idx
                            )
                        );
                    }
                    end.accept_mutate(&mut subs_expr);
                    Stmt::For(
                        For::make(
                            iter_vars[idx].borrow().var(),
                            iter_vars[idx].borrow().start(),
                            subs_expr.expr(),
                            iter_vars[idx].borrow().step(),
                            build_recursive(idx + 1, stage, iter_vars)
                        )
                    )
                }
            }
        }
    }

    build_recursive(0, stage, iter_vars)
}

#[rustfmt::skip]
pub fn build_nested_for2(stage: RcMut<Stage>) -> Stmt {
    let mut axes = stage.borrow().leaf_id
        .borrow()
        .iter()
        .map(|(node_ptr, id)| { (stage.borrow().address_map.borrow()[&*node_ptr].clone(), *id) })
        .collect::<Vec<_>>();
    axes.sort_by(|a, b| a.1.cmp(&b.1));
    let axes = axes
        .iter()
        .map(|(node, _)| node.clone())
        .collect::<Vec<_>>();
    build_nested_for_helper(stage, &axes)
}
