use std::{ cell::{ RefCell, RefMut }, collections::{ HashMap, HashSet } };

use quote::ToTokens;

use super::{ dag::{ Graph, Graph2, Var, Var2 }, kernel_type::KernelType, node::Node };

pub(crate) fn fuse<'ast>(candidates: &'ast Graph<'ast>) -> Vec<HashSet<Var2>> {
    let unfused = RefCell::new(candidates.to_graph2());
    let mut results = Vec::new();
    while let Some(next) = yield_candidate(unfused.borrow_mut()) {
        let mut block = HashSet::new();
        match next {
            Node::Unary(unary, _) => {
                block.insert(Var2 { ident: unary.output.clone() });
                let kernel_type = KernelType::Unary;
                for succ in children(&next, candidates) {
                    fuse_children(&succ, kernel_type, &mut block, candidates);
                }
                for pred in parents(&next, candidates) {
                    fuse_parents(&pred, kernel_type, &mut block, candidates);
                }
            }
            Node::Binary(binary, _) => {
                block.insert(Var2 { ident: binary.output.clone() });
                let kernel_type = KernelType::Binary;
                for succ in children(&next, candidates) {
                    fuse_children(&succ, kernel_type, &mut block, candidates);
                }
                for pred in parents(&next, candidates) {
                    fuse_parents(&pred, kernel_type, &mut block, candidates);
                }
            }
        }
        let b_clone = block.clone();
        b_clone.iter().for_each(|node| {
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
            Node::Unary(_, _) => true,
            _ => false,
        }
    });
    match uary {
        None =>
            unfused_candidates.map
                .iter()
                .find(|(_, node)| {
                    match node {
                        Node::Unary(_, _) => false,
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
                Node::Unary(node, _) => {
                    block.insert(Var2 { ident: node.output.clone() });
                }
                Node::Binary(node, _) => {
                    block.insert(Var2 { ident: node.output.clone() });
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
    };
    Ok(kernel_type.infer_suc_kernel(&next_kernel_type))
}

pub fn parents<'a, 'ast>(node: &Node<'ast>, graph: &'a Graph<'ast>) -> HashSet<&'a Node<'ast>> {
    let mut parents = HashSet::new();
    match node {
        Node::Unary(unary, _) => {
            if let Some(parent) = graph.map.get(&(Var { ident: &unary.operand })) {
                parents.insert(*parent);
            }
        }
        Node::Binary(binary, _) => {
            if let Some(parent) = graph.map.get(&(Var { ident: &binary.left })) {
                parents.insert(*parent);
            }
            if let Some(parent) = graph.map.get(&(Var { ident: &binary.right })) {
                parents.insert(*parent);
            }
        }
    }
    parents
}

pub fn children<'a, 'ast>(node: &Node<'ast>, graph: &'a Graph<'ast>) -> HashSet<&'a Node<'ast>> {
    match node {
        Node::Unary(unary, _) => {
            graph.map
                .iter()
                .filter(|(_, node)| {
                    match node {
                        Node::Unary(u, _) =>
                            u.operand.to_token_stream().to_string() ==
                                unary.output.to_token_stream().to_string(),
                        Node::Binary(binary, _) =>
                            binary.left == unary.output || binary.right == unary.output,
                    }
                })
                .map(|(_, node)| *node)
                .collect()
        }
        Node::Binary(binary, _) => {
            graph.map
                .iter()
                .filter(|(_, node)| {
                    match node {
                        Node::Unary(u, _) =>
                            u.operand.to_token_stream().to_string() ==
                                binary.output.to_token_stream().to_string(),
                        Node::Binary(binary, _) =>
                            binary.left == binary.output || binary.right == binary.output,
                    }
                })
                .map(|(_, node)| *node)
                .collect()
        }
    }
}
