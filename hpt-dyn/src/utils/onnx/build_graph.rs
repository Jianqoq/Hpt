use std::collections::HashMap;

use petgraph::prelude::StableGraph;

use super::operators::Operator;

pub(super) fn build_graph(
    operators: &[Operator],
    tensor_to_node: &HashMap<&str, &str>
) -> StableGraph<Operator, ()> {
    let mut stablegraph = StableGraph::new();
    let mut node_indices = HashMap::new();
    for node in operators.iter() {
        let idx = stablegraph.add_node(node.clone());
        node_indices.insert(node.id(), idx);
    }
    for node in operators.iter() {
        let current_idx = node_indices[node.id()];
        for input in node.inputs() {
            if let Some(b) = tensor_to_node.get(input) {
                stablegraph.add_edge(
                    *node_indices
                        .get(b)
                        .expect(&format!("node {} not found", tensor_to_node[input])),
                    current_idx,
                    ()
                );
            }
        }
    }

    stablegraph
}
