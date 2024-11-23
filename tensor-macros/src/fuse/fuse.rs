use std::{ cell::{ RefCell, RefMut }, collections::{ HashMap, HashSet } };

use petgraph::graph::NodeIndex;
use quote::ToTokens;

use super::{ dag::{ Graph, Graph2, _Graph }, kernel_type::KernelType, node::Node };

#[derive(Debug)]
pub(crate) struct FusionGroup {
    pub(crate) vars: Vec<HashSet<NodeIndex>>,
}

pub(crate) fn fuse_graph<'ast>(graph: &'ast petgraph::Graph<&crate::fuse::node::Node<'ast>, ()>) -> FusionGroup {
    fuse(&graph)
}

pub(crate) fn out_degree<'ast>(graph: &'ast _Graph<'ast>) -> HashMap<syn::Ident, usize> {
    let mut out_degree = HashMap::new();
    let mut current_graph = Some(graph);
    while let Some(next_graph) = current_graph {
        _out_degree(&next_graph.map, &mut out_degree);
        if let Some(next_graph) = &next_graph.next_graph {
            current_graph = Some(next_graph);
        } else {
            break;
        }
    }
    out_degree
}

pub(crate) fn _out_degree<'ast>(
    map: &'ast HashMap<&'ast syn::Ident, &'ast Node<'ast>>,
    out_degree: &mut HashMap<syn::Ident, usize>
) {
    for out in map.keys() {
        let mut degree = 0;
        for node in map.values() {
            match node {
                Node::Unary(unary) => {
                    if &unary.operand == *out {
                        degree += 1;
                    }
                }
                Node::Binary(binary) => {
                    if &binary.left == *out || &binary.right == *out {
                        degree += 1;
                    }
                }
                Node::Input(_) => {}
            }
        }
        out_degree
            .entry((*out).clone())
            .and_modify(|d| {
                *d += degree;
            })
            .or_insert(degree);
    }
}

pub(crate) fn fuse<'ast>(
    candidates: &'ast petgraph::Graph<&crate::fuse::node::Node<'ast>, ()>
) -> FusionGroup {
    let mut unfused = candidates.clone();

    let mut results = Vec::new();
    while let Some(idx) = yield_candidate(&mut unfused) {
        let mut block = HashSet::new();
        match unfused.node_weight(idx).expect("node weight not found") {
            Node::Unary(_) => {
                block.insert(idx);
                let kernel_type = KernelType::Unary;
                for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                    fuse_children(succ, kernel_type, &mut block, &unfused);
                }
                for pred in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
                    fuse_parents(pred, kernel_type, &mut block, &unfused);
                }
            }
            Node::Binary(_) => {
                block.insert(idx);
                let kernel_type = KernelType::Binary;
                for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                    fuse_children(succ, kernel_type, &mut block, &unfused);
                }
                for pred in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
                    fuse_parents(pred, kernel_type, &mut block, &unfused);
                }
            }
            Node::Input(_) => {
                block.insert(idx);
            }
        }
        let mut block_vec = block.iter().collect::<Vec<_>>();
        block_vec.sort_by(|a, b| b.cmp(a));
        block_vec.into_iter().for_each(|node| {
            unfused.remove_node(*node);
        });
        results.push(block);
    }
    let ret = FusionGroup {
        vars: results,
    };
    ret
}

pub(crate) fn yield_candidate<'a, 'ast>(
    unfused_candidates: &mut petgraph::Graph<&Node<'ast>, ()>
) -> Option<NodeIndex> {
    let unary = unfused_candidates.node_indices().find(|x| {
        match unfused_candidates.node_weight(*x) {
            Some(Node::Unary(_)) => true,
            _ => false,
        }
    });
    match unary {
        None =>
            unfused_candidates
                .node_indices()
                .find(|x| {
                    match unfused_candidates.node_weight(*x) {
                        _ => true,
                    }
                })
                .map(|x| x),
        Some(node) => Some(node),
    }
}

pub fn fuse_parents<'ast>(
    pred: NodeIndex,
    next_kernel_type: KernelType,
    block: &mut HashSet<NodeIndex>,
    graph: &'ast petgraph::Graph<&Node<'ast>, ()>
) {
    match
        pred_kernel_fusable(
            next_kernel_type,
            graph.node_weight(pred).expect("node weight not found")
        )
    {
        Ok(Some(kernel_type)) => {
            match graph.node_weight(pred).expect("node weight not found") {
                Node::Unary(_) | Node::Binary(_) | Node::Input(_) => {
                    block.insert(pred);
                }
            }
            for next in graph.neighbors_directed(pred, petgraph::Direction::Incoming) {
                fuse_parents(next, kernel_type, block, graph);
            }
        }
        Ok(None) => {}
        Err(_) => {}
    }
}

pub fn fuse_children<'ast>(
    succ: NodeIndex,
    prev_kernel_type: KernelType,
    block: &mut HashSet<NodeIndex>,
    graph: &'ast petgraph::Graph<&Node<'ast>, ()>
) {
    match
        suc_kernel_fusable(
            prev_kernel_type,
            graph.node_weight(succ).expect("node weight not found")
        )
    {
        Ok(Some(kernel_type)) => {
            match graph.node_weight(succ).expect("node weight not found") {
                Node::Unary(_) | Node::Binary(_) | Node::Input(_) => {
                    block.insert(succ);
                }
            }
            for next in graph.neighbors_directed(succ, petgraph::Direction::Outgoing) {
                fuse_children(next, kernel_type, block, graph);
            }
        }
        Ok(None) => {}
        Err(_) => {}
    }
}

pub fn pred_kernel_fusable<'ast>(
    next_kernel_type: KernelType,
    pred: &Node<'ast>
) -> anyhow::Result<Option<KernelType>> {
    let pred_kernel_type = match pred {
        Node::Unary(..) => KernelType::Unary,
        Node::Binary(..) => KernelType::Binary,
        Node::Input(..) => KernelType::Unary,
    };
    Ok(pred_kernel_type.infer_pred_kernel(&next_kernel_type))
}

pub fn suc_kernel_fusable<'ast>(
    kernel_type: KernelType,
    next: &Node<'ast>
) -> anyhow::Result<Option<KernelType>> {
    let next_kernel_type = match next {
        Node::Unary(..) => KernelType::Unary,
        Node::Binary(..) => KernelType::Binary,
        Node::Input(..) => KernelType::Unary,
    };
    Ok(kernel_type.infer_suc_kernel(&next_kernel_type))
}

pub fn parents<'a, 'ast>(node: &Node<'ast>, graph: &'a _Graph<'ast>) -> HashSet<&'a Node<'ast>> {
    let mut parents = HashSet::new();
    match node {
        Node::Unary(unary) => {
            if let Some(parent) = graph.map.get(&unary.operand) {
                parents.insert(*parent);
            }
        }
        Node::Binary(binary) => {
            if let Some(parent) = graph.map.get(&binary.left) {
                parents.insert(*parent);
            }
            if let Some(parent) = graph.map.get(&binary.right) {
                parents.insert(*parent);
            }
        }
        Node::Input(..) => {}
    }
    parents
}
