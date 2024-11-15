use std::collections::{ HashMap, HashSet, VecDeque };

use super::{ edges::Edges, node::Node, visitor::_Visitor };

#[derive(Debug)]
pub(crate) struct Graph<'ast> {
    pub(crate) _graph: _Graph<'ast>,
}

impl<'ast> Graph<'ast> {
    pub(crate) fn from_visitor(visitor: &'ast _Visitor<'ast>) -> Self {
        let mut _graph = _Graph::from_nodes(&visitor.nodes);
        if let Some(next_visitor) = &visitor.next_visitor {
            _graph.next_graph = Some(Box::new(Graph::from_visitor(&next_visitor)._graph));
        }
        Self { _graph }
    }
}

#[derive(Clone)]
pub(crate) struct _Graph<'ast> {
    pub(crate) map: HashMap<&'ast syn::Ident, &'ast Node<'ast>>,
    pub(crate) edges: Edges<'ast>,
    pub(crate) next_graph: Option<Box<_Graph<'ast>>>,
}

impl<'ast> std::fmt::Debug for _Graph<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[derive(Eq, PartialEq, Hash)]
        struct Var<'ast>(&'ast syn::Ident);
        impl<'ast> std::fmt::Debug for Var<'ast> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&format!("{}", self.0.to_string()))
            }
        }
        let map = self.map
            .iter()
            .map(|(ident, node)| { (Var(ident), node) })
            .collect::<HashMap<_, _>>();
        if let Some(next_graph) = &self.next_graph {
            f.debug_struct("Graph").field("map", &map).field("next_graph", &next_graph).finish()
        } else {
            f.debug_struct("Graph").field("map", &map).finish()
        }
    }
}

impl<'ast> _Graph<'ast> {
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
                    map.insert(&input, node);
                }
            }
        }
        let mut edges = HashMap::new();
        for &node_id in map.keys() {
            if let Some(neighbors) = map.get(node_id) {
                match neighbors {
                    Node::Unary(unary) => {
                        edges.entry(node_id).or_insert(HashSet::new()).insert(&unary.operand);
                    }
                    Node::Binary(binary) => {
                        edges.entry(node_id).or_insert(HashSet::new()).insert(&binary.left);
                        edges.entry(node_id).or_insert(HashSet::new()).insert(&binary.right);
                    }
                    Node::Input(_) => {
                        edges.entry(node_id).or_insert(HashSet::new());
                    }
                }
            }
        }
        Self {
            map,
            edges: Edges::from(edges),
            next_graph: None,
        }
    }

    pub(crate) fn to_graph2(&self) -> Graph2<'ast> {
        let mut map = HashMap::new();
        for (&k, v) in self.map.iter() {
            map.insert(k.clone(), *v);
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
    pub(crate) map: HashMap<syn::Ident, &'ast Node<'ast>>,
}
