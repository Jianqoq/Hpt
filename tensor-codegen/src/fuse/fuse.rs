use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use super::{ kernel_type::KernelType, node::Node };
#[derive(Eq, Hash, PartialEq)]
pub(crate) struct Input {
    pub(crate) var: syn::Ident,
    pub(crate) stmt_index: i64,
    pub(crate) block_idx: usize,
    pub(crate) comp_graph_idx: NodeIndex,
}

impl std::fmt::Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("var", &self.var.to_string())
            .field("stmt_index", &self.stmt_index)
            .field("block_idx", &self.block_idx)
            .field("comp_graph_idx", &self.comp_graph_idx)
            .finish()
    }
}

#[derive(Eq, Hash, PartialEq)]
pub(crate) struct Output {
    pub(crate) var: syn::Ident,
    pub(crate) stmt_index: i64,
    pub(crate) block_idx: usize,
    pub(crate) comp_graph_idx: NodeIndex,
}

impl std::fmt::Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Output")
            .field("var", &self.var.to_string())
            .field("stmt_index", &self.stmt_index)
            .field("block_idx", &self.block_idx)
            .field("comp_graph_idx", &self.comp_graph_idx)
            .finish()
    }
}

pub(crate) struct FusionGroup {
    pub(crate) vars: Vec<(HashSet<NodeIndex>, HashSet<Input>, HashSet<Output>)>,
}

impl std::fmt::Debug for FusionGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusionGroup").field("vars", &self.vars).finish()
    }
}

pub(crate) fn fuse<'ast>(
    cfg: &crate::fuse::cfg::CFG,
    candidates: &'ast petgraph::stable_graph::StableGraph<
        &(crate::fuse::node::Node, i64, usize),
        ()
    >
) -> FusionGroup {
    let mut unfused = candidates.clone();
    let edges = candidates.edge_indices();
    let mut edges_map = HashSet::new();
    for edge in edges {
        edges_map.insert(edge);
    }
    let mut results = Vec::new();
    let mut to_remove = Vec::new();
    let mut old_block = HashSet::new();
    while let Some(idx) = yield_candidate(&mut unfused) {
        let mut block = HashSet::new();
        let mut inputs = HashSet::new();
        let mut outputs = HashSet::new();
        match unfused.node_weight(idx).expect("node weight not found") {
            (Node::Unary(unary), stmt_index, block_idx) => {
                let basic_block = cfg.graph
                    .node_weight(NodeIndex::new(*block_idx))
                    .expect("node weight not found");
                block.insert(idx);
                let kernel_type = KernelType::Unary;
                if !basic_block.live_out.contains(&unary.output) {
                    for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                        fuse_children(
                            succ,
                            kernel_type,
                            &mut block,
                            &unfused,
                            basic_block,
                            &mut outputs
                        );
                    }
                } else {
                    outputs.insert(Output {
                        var: unary.output.clone(),
                        stmt_index: *stmt_index,
                        block_idx: *block_idx,
                        comp_graph_idx: idx,
                    });
                }
                for pred in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
                    let (var, stmt_index, block_idx) = match
                        unfused.node_weight(pred).expect("node weight not found")
                    {
                        (Node::Unary(unary), stmt_index, block_idx) =>
                            (&unary.output, stmt_index, block_idx),
                        (Node::Binary(binary), stmt_index, block_idx) =>
                            (&binary.output, stmt_index, block_idx),
                        (Node::Input(input), stmt_index, block_idx) =>
                            (input, stmt_index, block_idx),
                    };
                    if basic_block.live_out.contains(var) {
                        block.insert(pred);
                        inputs.insert(Input {
                            var: var.clone(),
                            stmt_index: *stmt_index,
                            block_idx: *block_idx,
                            comp_graph_idx: pred,
                        });
                    } else {
                        fuse_parents(
                            pred,
                            kernel_type,
                            &mut block,
                            &unfused,
                            basic_block,
                            &mut inputs
                        );
                    }
                }
            }
            (Node::Binary(binary), stmt_index, block_idx) => {
                let basic_block = cfg.graph
                    .node_weight(NodeIndex::new(*block_idx))
                    .expect("node weight not found");
                block.insert(idx);
                let kernel_type = KernelType::Binary;
                if !basic_block.live_out.contains(&binary.output) {
                    for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                        fuse_children(
                            succ,
                            kernel_type,
                            &mut block,
                            &unfused,
                            basic_block,
                            &mut outputs
                        );
                    }
                } else {
                    outputs.insert(Output {
                        var: binary.output.clone(),
                        stmt_index: *stmt_index,
                        block_idx: *block_idx,
                        comp_graph_idx: idx,
                    });
                }
                for pred in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
                    let (var, stmt_index, block_idx) = match
                        unfused.node_weight(pred).expect("node weight not found")
                    {
                        (Node::Unary(unary), stmt_index, block_idx) =>
                            (&unary.output, stmt_index, block_idx),
                        (Node::Binary(binary), stmt_index, block_idx) =>
                            (&binary.output, stmt_index, block_idx),
                        (Node::Input(input), stmt_index, block_idx) =>
                            (input, stmt_index, block_idx),
                    };
                    if basic_block.live_out.contains(var) {
                        block.insert(pred);
                        inputs.insert(Input {
                            var: var.clone(),
                            stmt_index: *stmt_index,
                            block_idx: *block_idx,
                            comp_graph_idx: pred,
                        });
                    } else {
                        fuse_parents(
                            pred,
                            kernel_type,
                            &mut block,
                            &unfused,
                            basic_block,
                            &mut inputs
                        );
                    }
                }
            }
            (Node::Input(inp), stmt_index, block_idx) => {
                block.insert(idx);
                let basic_block = cfg.graph
                    .node_weight(NodeIndex::new(*block_idx))
                    .expect("node weight not found");
                let kernel_type = KernelType::Unary;
                if !basic_block.live_out.contains(&inp) {
                    for succ in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                        fuse_children(
                            succ,
                            kernel_type,
                            &mut block,
                            &unfused,
                            basic_block,
                            &mut outputs
                        );
                    }
                } else {
                    outputs.insert(Output {
                        var: inp.clone(),
                        stmt_index: *stmt_index,
                        block_idx: *block_idx,
                        comp_graph_idx: idx,
                    });
                }
            }
        }
        to_remove.clear();
        for edge in unfused.edge_indices().collect::<Vec<_>>() {
            let edge_endpoint = unfused.edge_endpoints(edge).expect("edge not found");
            if block.contains(&edge_endpoint.0) && block.contains(&edge_endpoint.1) {
                to_remove.push(edge.clone());
                unfused.remove_edge(edge).expect("remove edge failed");
            }
        }
        if old_block == block {
            for node in block.iter() {
                unfused.remove_node(*node);
            }
            old_block.clear();
        } else {
            old_block = block.clone();
            block.iter().for_each(|node| {
                let out_goings = unfused
                    .neighbors_directed(*node, petgraph::Direction::Outgoing)
                    .count();
                let incomings = unfused
                    .neighbors_directed(*node, petgraph::Direction::Incoming)
                    .count();
                if out_goings == 0 && incomings == 0 {
                    unfused.remove_node(*node);
                }
            });
            results.push((block, inputs, outputs));
        }
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
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>,
    basic_block: &crate::fuse::cfg::BasicBlock,
    inputs: &mut HashSet<Input>
) {
    match
        pred_kernel_fusable(
            next_kernel_type,
            graph.node_weight(pred).expect("node weight not found")
        )
    {
        Ok(Some(kernel_type)) => {
            block.insert(pred);
            match graph.node_weight(pred).expect("node weight not found") {
                (Node::Unary(unary), stmt_index, block_idx) => {
                    let incomings = graph.neighbors_directed(pred, petgraph::Direction::Incoming);
                    if incomings.count() == 0 {
                        inputs.insert(Input {
                            var: unary.output.clone(),
                            stmt_index: *stmt_index,
                            block_idx: *block_idx,
                            comp_graph_idx: pred,
                        });
                    } else {
                        for next in graph.neighbors_directed(pred, petgraph::Direction::Incoming) {
                            fuse_parents(next, kernel_type, block, graph, basic_block, inputs);
                        }
                    }
                }
                (Node::Binary(binary), stmt_index, block_idx) => {
                    let incomings = graph.neighbors_directed(pred, petgraph::Direction::Incoming);
                    if incomings.count() == 0 {
                        inputs.insert(Input {
                            var: binary.output.clone(),
                            stmt_index: *stmt_index,
                            block_idx: *block_idx,
                            comp_graph_idx: pred,
                        });
                    } else {
                        for next in graph.neighbors_directed(pred, petgraph::Direction::Incoming) {
                            fuse_parents(next, kernel_type, block, graph, basic_block, inputs);
                        }
                    }
                }
                (Node::Input(input), stmt_index, block_idx) => {
                    inputs.insert(Input {
                        var: input.clone(),
                        stmt_index: *stmt_index,
                        block_idx: *block_idx,
                        comp_graph_idx: pred,
                    });
                }
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
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>,
    basic_block: &crate::fuse::cfg::BasicBlock,
    outputs: &mut HashSet<Output>
) {
    match
        suc_kernel_fusable(
            prev_kernel_type,
            graph.node_weight(succ).expect("node weight not found")
        )
    {
        Ok(Some(kernel_type)) => {
            block.insert(succ);
            match graph.node_weight(succ).expect("node weight not found") {
                (Node::Unary(unary), stmt_index, block_idx) => {
                    let out_going = graph.neighbors_directed(succ, petgraph::Direction::Outgoing);
                    if !basic_block.live_out.contains(&unary.output) && out_going.count() > 0 {
                        for next in graph.neighbors_directed(succ, petgraph::Direction::Outgoing) {
                            fuse_children(next, kernel_type, block, graph, basic_block, outputs);
                        }
                    } else {
                        outputs.insert(Output {
                            var: unary.output.clone(),
                            stmt_index: *stmt_index,
                            block_idx: *block_idx,
                            comp_graph_idx: succ,
                        });
                    }
                }
                (Node::Binary(binary), stmt_index, block_idx) => {
                    let out_going = graph.neighbors_directed(succ, petgraph::Direction::Outgoing);
                    if !basic_block.live_out.contains(&binary.output) && out_going.count() > 0 {
                        for next in graph.neighbors_directed(succ, petgraph::Direction::Outgoing) {
                            fuse_children(next, kernel_type, block, graph, basic_block, outputs);
                        }
                    } else {
                        outputs.insert(Output {
                            var: binary.output.clone(),
                            stmt_index: *stmt_index,
                            block_idx: *block_idx,
                            comp_graph_idx: succ,
                        });
                    }
                }
                _ => {}
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
