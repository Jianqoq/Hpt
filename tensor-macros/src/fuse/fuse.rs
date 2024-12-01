use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use super::{ kernel_type::KernelType, node::Node };

#[derive(Debug)]
pub(crate) struct FusionGroup {
    pub(crate) vars: Vec<HashSet<NodeIndex>>,
}

pub(crate) fn fuse<'ast>(
    cfg: &crate::fuse::cfg::CFG,
    candidates: &'ast petgraph::stable_graph::StableGraph<
        &(crate::fuse::node::Node, i64, usize),
        ()
    >
) -> FusionGroup {
    let mut unfused = candidates.clone();

    let mut results = Vec::new();
    while let Some(idx) = yield_candidate(&mut unfused) {
        let mut block = HashSet::new();
        match unfused.node_weight(idx).expect("node weight not found") {
            (Node::Unary(unary), _, block_idx) => {
                let basic_block = cfg.graph
                    .node_weight(NodeIndex::new(*block_idx))
                    .expect("node weight not found");
                block.insert(idx);
                let kernel_type = KernelType::Unary;
                if !basic_block.live_out.contains(&unary.output) {
                    for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                        fuse_children(succ, kernel_type, &mut block, &unfused);
                    }
                }
                for pred in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
                    let var = match unfused.node_weight(pred).expect("node weight not found") {
                        (Node::Unary(unary), _, _) => &unary.output,
                        (Node::Binary(binary), _, _) => &binary.output,
                        (Node::Input(input), _, _) => input,
                    };
                    if !basic_block.live_out.contains(var) {
                        fuse_parents(pred, kernel_type, &mut block, &unfused);
                    }
                }
            }
            (Node::Binary(binary), _, block_idx) => {
                let basic_block = cfg.graph
                    .node_weight(NodeIndex::new(*block_idx))
                    .expect("node weight not found");
                block.insert(idx);
                let kernel_type = KernelType::Binary;
                if !basic_block.live_out.contains(&binary.output) {
                    for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                        fuse_children(succ, kernel_type, &mut block, &unfused);
                    }
                }
                for pred in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
                    let var = match unfused.node_weight(pred).expect("node weight not found") {
                        (Node::Unary(unary), _, _) => &unary.output,
                        (Node::Binary(binary), _, _) => &binary.output,
                        (Node::Input(input), _, _) => input,
                    };
                    if !basic_block.live_out.contains(var) {
                        fuse_parents(pred, kernel_type, &mut block, &unfused);
                    }
                }
            }
            (Node::Input(_), _, _) => {
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

pub(crate) fn yield_candidate<'a>(
    unfused_candidates: &mut petgraph::stable_graph::StableGraph<
        &(crate::fuse::node::Node, i64, usize),
        ()
    >
) -> Option<NodeIndex> {
    let unary = unfused_candidates.node_indices().find(|x| {
        match unfused_candidates.node_weight(*x) {
            Some((Node::Unary(_), _, _)) => true,
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

pub fn fuse_parents(
    pred: NodeIndex,
    next_kernel_type: KernelType,
    block: &mut HashSet<NodeIndex>,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>
) {
    match
        pred_kernel_fusable(
            next_kernel_type,
            graph.node_weight(pred).expect("node weight not found")
        )
    {
        Ok(Some(kernel_type)) => {
            match graph.node_weight(pred).expect("node weight not found") {
                (Node::Unary(_), _, _) | (Node::Binary(_), _, _) | (Node::Input(_), _, _) => {
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

pub fn fuse_children(
    succ: NodeIndex,
    prev_kernel_type: KernelType,
    block: &mut HashSet<NodeIndex>,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>
) {
    match
        suc_kernel_fusable(
            prev_kernel_type,
            graph.node_weight(succ).expect("node weight not found")
        )
    {
        Ok(Some(kernel_type)) => {
            match graph.node_weight(succ).expect("node weight not found") {
                (Node::Unary(_), _, _) | (Node::Binary(_), _, _) | (Node::Input(_), _, _) => {
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

pub fn pred_kernel_fusable(
    next_kernel_type: KernelType,
    pred: &(crate::fuse::node::Node, i64, usize)
) -> anyhow::Result<Option<KernelType>> {
    let pred_kernel_type = match pred {
        (Node::Unary(..), _, _) => KernelType::Unary,
        (Node::Binary(..), _, _) => KernelType::Binary,
        (Node::Input(..), _, _) => KernelType::Unary,
    };
    Ok(pred_kernel_type.infer_pred_kernel(&next_kernel_type))
}

pub fn suc_kernel_fusable(
    kernel_type: KernelType,
    next: &(crate::fuse::node::Node, i64, usize)
) -> anyhow::Result<Option<KernelType>> {
    let next_kernel_type = match next {
        (Node::Unary(..), _, _) => KernelType::Unary,
        (Node::Binary(..), _, _) => KernelType::Binary,
        (Node::Input(..), _, _) => KernelType::Unary,
    };
    Ok(kernel_type.infer_suc_kernel(&next_kernel_type))
}
