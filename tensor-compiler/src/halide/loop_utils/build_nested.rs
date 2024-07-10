use std::{ collections::VecDeque, sync::Arc };

use hashbrown::{ HashMap, HashSet };

use crate::{
    edges::Edges,
    halide::{ for_stmt::For, printer::IRPrinter, seq_stmt::Seq, stmt::Stmt },
    hlir::schedule::schedule::{ Node, RcMut, Stage },
    iter_var::IterVar,
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

pub fn build_nested_for_helper<T: Into<Stmt>>(
    stage: RcMut<Stage>,
    iter_vars: &[RcMut<Node>],
    main_stmt: T
) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(
        idx: usize,
        stage: RcMut<Stage>,
        iter_vars: &[RcMut<Node>],
        main_stmt: T
    ) -> Stmt {
        if idx == iter_vars.len() {
            main_stmt.into()
        } else {
            let mut seq = None;
            if
                let Some(stages) = stage
                    .borrow()
                    .attached_stage.borrow()
                    .get(&(iter_vars[idx].as_ptr() as usize))
            {
                let mut cnt = 0;
                let mut nodes_id = HashMap::new();
                let mut id_nodes = HashMap::new();

                let mut stages = stages.iter().map(|x| x.clone()).collect::<Vec<_>>();
                stages.push(stage.clone());

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

                for id in sorted.iter() {
                    let name = &id_nodes[id];
                    for i in stages.iter() {
                        if i.borrow().name == *name {
                            let halide = i.borrow().to_halid();
                            vec.push(halide);
                        }
                    }
                }

                seq = Some(vec);
            }
            let to_add = if let Some(mut seq) = seq {
                seq.push(build_recursive(idx + 1, stage, iter_vars, main_stmt));
                For::make(
                    iter_vars[idx].borrow().var(),
                    iter_vars[idx].borrow().start(),
                    iter_vars[idx].borrow().end(),
                    iter_vars[idx].borrow().step(),
                    Seq::make(seq)
                )
            } else {
                For::make(
                    iter_vars[idx].borrow().var(),
                    iter_vars[idx].borrow().start(),
                    iter_vars[idx].borrow().end(),
                    iter_vars[idx].borrow().step(),
                    build_recursive(idx + 1, stage, iter_vars, main_stmt)
                )
            };
            Stmt::For(to_add)
        }
    }

    build_recursive(0, stage, iter_vars, main_stmt)
}

#[rustfmt::skip]
pub fn build_nested_for2(stage: RcMut<Stage>, main_stmt: Stmt) -> Stmt {
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
    build_nested_for_helper(stage, &axes, main_stmt)
}
