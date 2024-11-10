use std::{ cell::{ RefCell, RefMut }, collections::HashSet };

use quote::ToTokens;

use super::{ dag::{ Graph, Graph2, Var, Var2 }, kernel_type::KernelType, node::Node };

pub(crate) fn fuse<'ast>(candidates: &'ast Graph<'ast>) -> Vec<HashSet<Var2>> {
    let unfused = RefCell::new(candidates.to_graph2());
    let mut results = Vec::new();
    while let Some(next) = yield_candidate(unfused.borrow_mut()) {
        let mut block = HashSet::new();
        match next {
            Node::Unary(unary) => {
                block.insert(Var2 { ident: unary.output.clone() });
                let kernel_type = KernelType::Unary;
                for succ in children(&next, candidates) {
                    fuse_children(&succ, kernel_type, &mut block, candidates);
                }
                for pred in parents(&next, candidates) {
                    fuse_parents(&pred, kernel_type, &mut block, candidates);
                }
            }
            Node::Binary(binary) => {
                block.insert(Var2 { ident: binary.output.clone() });
                let kernel_type = KernelType::Binary;
                for succ in children(&next, candidates) {
                    fuse_children(&succ, kernel_type, &mut block, candidates);
                }
                for pred in parents(&next, candidates) {
                    fuse_parents(&pred, kernel_type, &mut block, candidates);
                }
            }
            Node::Input(input) => {
                block.insert(Var2 { ident: input.ident.clone() });
            }
        }
        block.iter().for_each(|node| {
            unfused.borrow_mut().map.remove(node);
        });
        results.push(block);
    }
    results
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
    block: &mut HashSet<Var2>,
    graph: &'ast Graph<'ast>
) {
    match pred_kernel_fusable(next_kernel_type, pred) {
        Ok(Some(kernel_type)) => {
            match pred {
                Node::Unary(unary) => {
                    block.insert(Var2 { ident: unary.output.clone() });
                }
                Node::Binary(binary) => {
                    block.insert(Var2 { ident: binary.output.clone() });
                }
                Node::Input(input) => {
                    block.insert(Var2 { ident: input.ident.clone() });
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
    block: &mut HashSet<Var2>,
    graph: &'ast Graph<'ast>
) {
    match suc_kernel_fusable(prev_kernel_type, succ) {
        Ok(Some(kernel_type)) => {
            match succ {
                Node::Unary(node) => {
                    block.insert(Var2 { ident: node.output.clone() });
                }
                Node::Binary(node) => {
                    block.insert(Var2 { ident: node.output.clone() });
                }
                Node::Input(input) => {
                    block.insert(Var2 { ident: input.ident.clone() });
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

pub fn parents<'a, 'ast>(node: &Node<'ast>, graph: &'a Graph<'ast>) -> HashSet<&'a Node<'ast>> {
    let mut parents = HashSet::new();
    match node {
        Node::Unary(unary) => {
            if let Some(parent) = graph.map.get(&(Var { ident: &unary.operand })) {
                parents.insert(*parent);
            }
        }
        Node::Binary(binary) => {
            if let Some(parent) = graph.map.get(&(Var { ident: &binary.left })) {
                parents.insert(*parent);
            }
            if let Some(parent) = graph.map.get(&(Var { ident: &binary.right })) {
                parents.insert(*parent);
            }
        }
        Node::Input(..) => {}
    }
    parents
}

pub fn children<'a, 'ast>(node: &Node<'ast>, graph: &'a Graph<'ast>) -> HashSet<&'a Node<'ast>> {
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
                        Node::Binary(bi) =>
                            bi.left == binary.output || bi.right == binary.output,
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
                                input.ident.to_token_stream().to_string(),
                        Node::Binary(bi) => &bi.left == input.ident || &bi.right == input.ident,
                        Node::Input(..) => false,
                    }
                })
                .map(|(_, node)| *node)
                .collect()
        }
    }
}
