use std::{ cell::{ RefCell, RefMut }, collections::{ HashMap, HashSet } };

use quote::ToTokens;

use super::{ dag::{ Graph, Graph2, _Graph }, kernel_type::KernelType, node::Node };

pub(crate) struct FusionGroup {
    pub(crate) vars: Vec<HashSet<syn::Ident>>,
    pub(crate) _next_group: Option<Box<FusionGroup>>,
}

impl std::fmt::Debug for FusionGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vars = self.vars
            .iter()
            .map(|v|
                v
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();
        if let Some(next_group) = &self._next_group {
            f.debug_struct("FusionGroup")
                .field("vars", &vars)
                .field("next_group", &next_group)
                .finish()
        } else {
            write!(f, "FusionGroup {{ vars: {:?} }}", vars)
        }
    }
}

pub(crate) fn fuse_graph<'ast>(graph: &'ast Graph<'ast>) -> FusionGroup {
    fuse(&graph._graph)
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

pub(crate) fn fuse<'ast>(candidates: &'ast _Graph<'ast>) -> FusionGroup {
    let out_degree = out_degree(candidates);
    // println!("out_degree: {:#?}", out_degree);

    let unfused = RefCell::new(candidates.to_graph2());

    let mut results = Vec::new();
    while let Some(node) = yield_candidate(unfused.borrow_mut()) {
        let mut block = HashSet::new();
        match node {
            Node::Unary(unary) => {
                block.insert(unary.output.clone());
                let kernel_type = KernelType::Unary;
                for succ in children(&node, candidates) {
                    fuse_children(&succ, kernel_type, &mut block, candidates);
                }
                for pred in parents(&node, candidates) {
                    fuse_parents(&pred, kernel_type, &mut block, candidates)
                }
            }
            Node::Binary(binary) => {
                block.insert(binary.output.clone());
                let kernel_type = KernelType::Binary;
                for succ in children(&node, candidates) {
                    fuse_children(&succ, kernel_type, &mut block, candidates);
                }
                for pred in parents(&node, candidates) {
                    fuse_parents(&pred, kernel_type, &mut block, candidates)
                }
            }
            Node::Input(input) => {
                block.insert(input.clone());
            }
        }
        block.iter().for_each(|node| {
            unfused.borrow_mut().map.remove(node);
        });
        results.push(block);
    }
    let mut ret = FusionGroup {
        vars: results,
        _next_group: None,
    };
    if let Some(next_graph) = &candidates.next_graph {
        ret._next_group = Some(Box::new(fuse(&next_graph)));
    }
    ret
}

pub(crate) fn yield_candidate<'a, 'ast>(
    unfused_candidates: RefMut<'a, Graph2<'ast>>
) -> Option<&'a Node<'ast>> {
    let uary = unfused_candidates.map.iter().find(|(_, node)| {
        match node {
            Node::Unary(_) => true,
            _ => false,
        }
    });
    match uary {
        None =>
            unfused_candidates.map
                .iter()
                .find(|(_, node)| {
                    match node {
                        Node::Unary(_) => false,
                        _ => true,
                    }
                })
                .map(|(_, node)| *node),
        Some((_, node)) => Some(*node),
    }
}

pub fn fuse_parents<'ast>(
    pred: &Node<'ast>,
    next_kernel_type: KernelType,
    block: &mut HashSet<syn::Ident>,
    graph: &'ast _Graph<'ast>
) {
    match pred_kernel_fusable(next_kernel_type, pred) {
        Ok(Some(kernel_type)) => {
            match pred {
                Node::Unary(unary) => {
                    block.insert(unary.output.clone());
                }
                Node::Binary(binary) => {
                    block.insert(binary.output.clone());
                }
                Node::Input(input) => {
                    block.insert(input.clone());
                }
            }
            for next in parents(pred, graph) {
                fuse_parents(next, kernel_type, block, graph);
            }
        }
        Ok(None) => {}
        Err(_) => {}
    }
}

pub fn fuse_children<'ast>(
    succ: &'ast Node<'ast>,
    prev_kernel_type: KernelType,
    block: &mut HashSet<syn::Ident>,
    graph: &'ast _Graph<'ast>
) {
    match suc_kernel_fusable(prev_kernel_type, succ) {
        Ok(Some(kernel_type)) => {
            match succ {
                Node::Unary(node) => {
                    block.insert(node.output.clone());
                }
                Node::Binary(node) => {
                    block.insert(node.output.clone());
                }
                Node::Input(input) => {
                    block.insert(input.clone());
                }
            }
            for next in children(succ, graph) {
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

pub fn children<'a, 'ast>(node: &Node<'ast>, graph: &'a _Graph<'ast>) -> HashSet<&'a Node<'ast>> {
    match node {
        Node::Unary(unary) => {
            graph.map
                .iter()
                .filter(|(_, node)| {
                    match node {
                        Node::Unary(u) =>
                            u.operand.to_token_stream().to_string() ==
                                unary.output.to_token_stream().to_string(),
                        Node::Binary(binary) =>
                            binary.left == unary.output || binary.right == unary.output,
                        Node::Input(..) => false,
                    }
                })
                .map(|(_, node)| *node)
                .collect()
        }
        Node::Binary(binary) => {
            graph.map
                .iter()
                .filter(|(_, node)| {
                    match node {
                        Node::Unary(u) =>
                            u.operand.to_token_stream().to_string() ==
                                binary.output.to_token_stream().to_string(),
                        Node::Binary(bi) => bi.left == binary.output || bi.right == binary.output,
                        Node::Input(..) => false,
                    }
                })
                .map(|(_, node)| *node)
                .collect()
        }
        Node::Input(input) => {
            graph.map
                .iter()
                .filter(|(_, node)| {
                    match node {
                        Node::Unary(u) =>
                            u.operand.to_token_stream().to_string() ==
                                input.to_token_stream().to_string(),
                        Node::Binary(bi) => &bi.left == input || &bi.right == input,
                        Node::Input(..) => false,
                    }
                })
                .map(|(_, node)| *node)
                .collect()
        }
    }
}
