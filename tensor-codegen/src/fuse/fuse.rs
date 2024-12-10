use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use super::{ build_graph::CmpNode, kernel_type::KernelType };
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
    pub(crate) groups: Vec<HashSet<NodeIndex>>,
    pub(crate) inputs: Vec<HashSet<Input>>,
    pub(crate) outputs: Vec<HashSet<Output>>,
    pub(crate) stmt_to_remove: Vec<Vec<i64>>,
}

impl std::fmt::Debug for FusionGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusionGroup")
            .field("groups", &self.groups)
            .field("inputs", &self.inputs)
            .field("intermediates", &self.stmt_to_remove)
            .field("outputs", &self.outputs)
            .finish()
    }
}

pub(crate) fn cmp_fuse<'ast>(
    cfg: &crate::fuse::cfg::CFG,
    candidates: &'ast petgraph::stable_graph::StableGraph<CmpNode, ()>
) -> FusionGroup {
    let mut unfused = candidates.clone();
    let edges = candidates.edge_indices();
    let mut edges_map = HashSet::new();
    for edge in edges {
        edges_map.insert(edge);
    }
    let mut ret = FusionGroup {
        groups: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        stmt_to_remove: Vec::new(),
    };
    while let Some(idx) = cmp_yield_candidate(&mut unfused) {
        let mut block = HashSet::new();

        let node = &candidates[idx];
        let basic_block = cfg.graph
            .node_weight(NodeIndex::new(node.block_idx))
            .expect("node weight not found");
        block.insert(idx);
        if !basic_block.live_out.contains(&node.ident) {
            for output in unfused.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                cmp_fuse_children(
                    output,
                    node.kernel_type,
                    &mut block,
                    &unfused,
                    basic_block,
                );
            }
        }

        for inp in unfused.neighbors_directed(idx, petgraph::Direction::Incoming) {
            cmp_fuse_parents(inp, node.kernel_type, &mut block, &unfused, basic_block);
        }

        block.iter().for_each(|node| {
            unfused.remove_node(*node);
        });
        ret.groups.push(block);
        ret.inputs.push(HashSet::new());
        ret.outputs.push(HashSet::new());
        ret.stmt_to_remove.push(Vec::new());
    }
    ret
}

pub(crate) fn cmp_yield_candidate<'a>(
    unfused_candidates: &mut petgraph::stable_graph::StableGraph<CmpNode, ()>
) -> Option<NodeIndex> {
    let unary = unfused_candidates
        .node_indices()
        .find(|x| { unfused_candidates[*x].kernel_type == KernelType::Unary });
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

pub fn cmp_fuse_parents(
    pred: NodeIndex,
    next_kernel_type: KernelType,
    block: &mut HashSet<NodeIndex>,
    unfused: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    basic_block: &crate::fuse::cfg::BasicBlock,
) {
    let node = &unfused
        .node_weight(pred)
        .expect(format!("node weight not found {:?}, ", pred).as_str());
    match cmp_pred_kernel_fusable(next_kernel_type, node.kernel_type) {
        Ok(Some(kernel_type)) => {
            block.insert(pred);
            if !unfused.neighbors_directed(pred, petgraph::Direction::Incoming).count() == 0 {
                for inp in unfused.neighbors_directed(pred, petgraph::Direction::Incoming) {
                    cmp_fuse_parents(inp, kernel_type, block, unfused, basic_block);
                }
            }
        }
        Ok(None) => {
            block.insert(pred);
        }
        Err(_) => {}
    }
}

pub fn cmp_fuse_children(
    succ: NodeIndex,
    prev_kernel_type: KernelType,
    block: &mut HashSet<NodeIndex>,
    unfused: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    basic_block: &crate::fuse::cfg::BasicBlock,
) {
    let node = &unfused
        .node_weight(succ)
        .expect(format!("node weight not found {:?}, ", succ).as_str());
    block.insert(succ);
    match
        cmp_suc_kernel_fusable(
            prev_kernel_type,
            unfused.node_weight(succ).expect("node weight not found").kernel_type
        )
    {
        Ok(Some(kernel_type)) => {
            if !basic_block.live_out.contains(&node.ident) {
                for output in unfused.neighbors_directed(succ, petgraph::Direction::Outgoing) {
                    cmp_fuse_children(output, kernel_type, block, unfused, basic_block);
                }
            }
        }
        Ok(None) => {}
        Err(_) => {}
    }
}

pub fn cmp_pred_kernel_fusable(
    next_kernel_type: KernelType,
    pred: KernelType
) -> anyhow::Result<Option<KernelType>> {
    Ok(pred.infer_pred_kernel(&next_kernel_type))
}

pub fn cmp_suc_kernel_fusable(
    kernel_type: KernelType,
    next_kernel_type: KernelType
) -> anyhow::Result<Option<KernelType>> {
    Ok(kernel_type.infer_suc_kernel(&next_kernel_type))
}
