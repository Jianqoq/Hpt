use std::{ collections::VecDeque, sync::Arc };

use hashbrown::{ HashMap, HashSet };

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
        let store = if idx == iter_vars.len() - 1 || iter_vars.len() == 0 {
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
            if idx == iter_vars.len() - 1 {
                return Stmt::For(
                    For::make(
                        iter_vars[idx].borrow().var(),
                        iter_vars[idx].borrow().start(),
                        iter_vars[idx].borrow().end(),
                        iter_vars[idx].borrow().step(),
                        Seq::make(seq)
                    )
                );
            } else {
                seq.push(build_recursive(idx + 1, stage, iter_vars));
                return Stmt::For(
                    For::make(
                        iter_vars[idx].borrow().var(),
                        iter_vars[idx].borrow().start(),
                        iter_vars[idx].borrow().end(),
                        iter_vars[idx].borrow().step(),
                        Seq::make(seq)
                    )
                );
            }
        } else {
            if idx == iter_vars.len() - 1 {
                return Stmt::For(
                    For::make(
                        iter_vars[idx].borrow().var(),
                        iter_vars[idx].borrow().start(),
                        iter_vars[idx].borrow().end(),
                        iter_vars[idx].borrow().step(),
                        store.unwrap()
                    )
                );
            } else {
                if iter_vars.len() == 0 {
                    return store.unwrap();
                } else {
                    return Stmt::For(
                        For::make(
                            iter_vars[idx].borrow().var(),
                            iter_vars[idx].borrow().start(),
                            iter_vars[idx].borrow().end(),
                            iter_vars[idx].borrow().step(),
                            build_recursive(idx + 1, stage, iter_vars)
                        )
                    );
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
