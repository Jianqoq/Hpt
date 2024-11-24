use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use super::{ fuse::FusionGroup, gen_fuse::GenFuse };

pub(crate) struct ToRemove {
    pub(crate) to_remove: Vec<HashSet<(NodeIndex, i64)>>,
}

impl std::fmt::Debug for ToRemove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.to_remove.iter().map(|i|i.iter().map(|(_, idx)| *idx))).finish()
    }
}

pub(crate) fn gen_to_remove(gen_fuse: &GenFuse, fuse_group: &FusionGroup, graph: &petgraph::Graph<&(crate::fuse::node::Node<'_>, i64), ()>) -> ToRemove {
    let to_remove = _gen_to_remove(&fuse_group.vars, &gen_fuse.fused_inputs, &gen_fuse.fused_outs, graph);
    let ret = ToRemove { to_remove };
    ret
}

pub(crate) fn _gen_to_remove(
    fused_group: &Vec<HashSet<NodeIndex>>,
    fused_inputs: &Vec<HashSet<(NodeIndex, i64)>>,
    fused_outs: &Vec<(NodeIndex, i64)>,
    graph: &petgraph::Graph<&(crate::fuse::node::Node<'_>, i64), ()>
) -> Vec<HashSet<(NodeIndex, i64)>> {
    let mut to_remove = Vec::new();
    for ((input, total), out) in fused_inputs
        .iter()
        .zip(fused_group.iter())
        .zip(fused_outs.iter()) {
        let mut intermediate = total
            .iter()
            .map(|i| (*i, graph.node_weight(*i).expect("fuse_impl::to_remove::graph.get_node_weight").1))
            .collect::<HashSet<_>>();
        for input in input {
            intermediate.remove(input);
        }
        intermediate.remove(out);
        to_remove.push(intermediate);
    }
    to_remove
}
