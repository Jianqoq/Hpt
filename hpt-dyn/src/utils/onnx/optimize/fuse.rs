use petgraph::{ prelude::StableGraph, visit::EdgeRef, Direction::{ Incoming, Outgoing } };

use crate::utils::onnx::operators::{ Base, Conv2dFused, ConvActivation, Operator };

pub(crate) fn fuse_conv_unary(stablegraph: &mut StableGraph<Operator, ()>) {
    let mut edges_to_add = Vec::new();
    let mut nodes_to_remove = Vec::new();

    let node_indices = stablegraph.node_indices().collect::<Vec<_>>();
    for node_idx in node_indices {
        macro_rules! fuse {
            ($base:expr, $activation:expr) => {
                {
                    let mut incomings = stablegraph.edges_directed(node_idx, Incoming);
                    let input = incomings.next().expect("unary node has no incoming edge");
                    assert!(incomings.next().is_none(), "unary node has more than one incoming edge");
                    let source = input.source();
                    if matches!(stablegraph[source], Operator::Conv2d(_)) {
                        let base_output = $base.base.output.clone();
                        if let Operator::Conv2d(conv) = stablegraph[source].clone() {
                            let conv_fused = Conv2dFused {
                                input: conv.base.input,
                                output: base_output,
                                kernel: conv.base.kernel,
                                bias: conv.base.bias,
                                pads: conv.base.pads,
                                strides: conv.base.strides,
                                dilations: conv.base.dilations,
                                group: conv.base.group,
                                activation: $activation,
                            };
                            let operator = Operator::Conv2dFused(Base {
                                base: conv_fused,
                                id: conv.id,
                            });
                            stablegraph[source] = operator;
                            for output in stablegraph.edges_directed(node_idx, Outgoing) {
                                let target = output.target();
                                edges_to_add.push((source, target));
                            }
                            nodes_to_remove.push(node_idx);
                        }
                    }
                }
            };
        }
        match &stablegraph[node_idx] {
            Operator::Relu(unary) => fuse!(unary, ConvActivation::Relu),
            Operator::LeakyRelu(unary) => fuse!(unary, ConvActivation::LeakyRelu),
            Operator::Sigmoid(unary) => fuse!(unary, ConvActivation::Sigmoid),
            Operator::Gelu(unary) => fuse!(unary, ConvActivation::Gelu),
            Operator::Tanh(unary) => fuse!(unary, ConvActivation::Tanh),
            _ => (),
        }
    }

    for (source, target) in edges_to_add {
        stablegraph.add_edge(source, target, ());
    }
    for node_idx in nodes_to_remove {
        log::debug!("remove_node: {}", stablegraph[node_idx].id());
        stablegraph.remove_node(node_idx);
    }
}
