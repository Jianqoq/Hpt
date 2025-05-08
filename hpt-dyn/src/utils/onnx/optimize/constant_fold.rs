use std::collections::HashMap;

use petgraph::{ graph::NodeIndex, prelude::{ Incoming, Outgoing, StableGraph }, visit::EdgeRef };

use crate::{ utils::onnx::operators::Operator, Tensor };

pub(crate) fn pre_transpose(
    stablegraph: &mut StableGraph<Operator, ()>,
    initializer_map: &mut HashMap<String, Tensor>
) {
    let mut edges_to_add = Vec::new();
    let mut nodes_to_remove = Vec::new();
    for node_idx in stablegraph.node_indices() {
        if let Operator::Transpose(permute) = &stablegraph[node_idx] {
            let mut in_coming = stablegraph.edges_directed(node_idx, Incoming);

            let mut func = |
                initializer_map: &mut HashMap<String, Tensor>,
                edges_to_add: &mut Vec<(NodeIndex, NodeIndex)>
            | {
                let permute_func = |initializer: &mut Tensor| {
                    *initializer = initializer
                        .permute(&permute.base.perm)
                        .expect("permute error")
                        .contiguous()
                        .expect("contiguous error");
                };
                if let Some(initializer) = initializer_map.get_mut(stablegraph[node_idx].id()) {
                    log::debug!("permute: {}", stablegraph[node_idx].id());
                    permute_func(initializer);
                } else if let Some(initializer) = initializer_map.get_mut(&permute.base.input) {
                    log::debug!("permute: {}", permute.base.input);
                    permute_func(initializer);
                } else {
                    return;
                }
                for output in stablegraph.edges_directed(node_idx, Outgoing) {
                    let target = output.target();
                    for input in stablegraph.edges_directed(target, Incoming) {
                        let source = input.source();
                        edges_to_add.push((source, target));
                    }
                }
                nodes_to_remove.push(node_idx);
            };
            if let Some(ic) = in_coming.next() {
                assert!(
                    in_coming.next().is_none(),
                    "transpose node has more than one incoming edge"
                );
                let source = ic.source();
                let source_incomings = stablegraph.edges_directed(source, Incoming);
                if source_incomings.count() == 0 {
                    func(initializer_map, &mut edges_to_add);
                } else {
                    return;
                }
            } else {
                func(initializer_map, &mut edges_to_add);
            }
        }
    }
    for (source, target) in edges_to_add {
        stablegraph.add_edge(source, target, ());
    }
    for node_idx in nodes_to_remove {
        stablegraph.remove_node(node_idx);
    }
}
