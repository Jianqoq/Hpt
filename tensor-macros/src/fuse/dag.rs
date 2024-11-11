use std::collections::{ HashMap, HashSet, VecDeque };

use quote::ToTokens;

use super::{ edges::Edges, node::Node };

#[derive(Eq, Hash, PartialEq, Clone)]
pub(crate) struct Var {
    pub(crate) ident: proc_macro2::Ident,
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ident.to_token_stream().to_string())
    }
}

#[derive(Clone)]
pub(crate) struct Graph<'ast> {
    pub(crate) map: HashMap<&'ast syn::Ident, &'ast Node<'ast>>,
    pub(crate) edges: Edges<'ast>,
}

impl<'ast> std::fmt::Debug for Graph<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph").field("map", &self.map).finish()
    }
}

impl<'ast> Graph<'ast> {
    pub(crate) fn from_nodes(nodes: &'ast Vec<Node<'ast>>) -> Self {
        let mut map = HashMap::new();
        for node in nodes {
            match node {
                Node::Unary(unary, ..) => {
                    map.insert(&unary.output, node);
                }
                Node::Binary(binary, ..) => {
                    map.insert(&binary.output, node);
                }
                Node::Input(input) => {
                    map.insert(&input.ident, node);
                }
            }
        }
        let mut edges = HashMap::new();
        for &node_id in map.keys() {
            if let Some(neighbors) = map.get(node_id) {
                match neighbors {
                    Node::Unary(unary) => {
                        edges
                            .entry(node_id)
                            .or_insert(HashSet::new())
                            .insert(&unary.operand);
                    }
                    Node::Binary(binary) => {
                        edges
                            .entry(node_id)
                            .or_insert(HashSet::new())
                            .insert(&binary.left);
                        edges
                            .entry(node_id)
                            .or_insert(HashSet::new())
                            .insert(&binary.right);
                    }
                    Node::Input(_) => {
                        edges.entry(node_id).or_insert(HashSet::new());
                    }
                }
            }
        }
        Self { map, edges: Edges::from(edges) }
    }

    pub(crate) fn to_graph2(&self) -> Graph2<'ast> {
        let mut map = HashMap::new();
        for (&k, v) in self.map.iter() {
            map.insert(Var { ident: k.clone() }, *v);
        }
        Graph2 { map }
    }

    /// Performs a topological sort on the graph.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<&'ast Node<'ast>>)` containing nodes in topologically sorted order.
    /// - `Err(String)` if a cycle is detected in the graph.
    pub(crate) fn topological_sort(&'ast self) -> Option<VecDeque<syn::Ident>> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut order = VecDeque::new();
        let edges = self.edges.invert();
        // calculate in degree
        for (&node_id, _) in self.map.iter() {
            in_degree.entry(node_id).or_insert(0);
            let edges = edges.get(node_id);
            if let Some(edges) = edges {
                for target in edges {
                    *in_degree.entry(target).or_insert(0) += 1;
                }
            }
        }
        // push nodes with in degree 0 to queue
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }
        // topological sort
        while let Some(node_id) = queue.pop_front() {
            order.push_back(node_id.clone());
            if let Some(_) = self.map.get(&node_id) {
                let edges = edges.get(&node_id);
                if let Some(edges) = edges {
                    for &target in edges {
                        let degree = in_degree.get_mut(target).expect("topological_sort::degree");
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(target);
                        }
                    }
                }
            }
        }
        // check if there is a cycle
        if order.len() == self.map.len() {
            Some(order)
        } else {
            None // cycle detected
        }
    }
}

#[derive(Clone)]
pub(crate) struct Graph2<'ast> {
    pub(crate) map: HashMap<Var, &'ast Node<'ast>>,
}
