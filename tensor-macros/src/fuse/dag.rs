use std::collections::HashMap;

use quote::ToTokens;

use super::node::Node;

#[derive(Eq, Hash, PartialEq, Clone)]
pub(crate) struct Var<'ast> {
    pub(crate) ident: &'ast proc_macro2::Ident,
}

#[derive(Eq, Hash, PartialEq, Clone)]
pub(crate) struct Var2 {
    pub(crate) ident: proc_macro2::Ident,
}

impl<'ast> std::fmt::Debug for Var<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ident.to_token_stream().to_string())
    }
}

impl<'ast> ToTokens for Var<'ast> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.ident.to_tokens(tokens);
    }
}

impl std::fmt::Debug for Var2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ident.to_token_stream().to_string())
    }
}

#[derive(Clone)]
pub(crate) struct Graph<'ast> {
    pub(crate) map: HashMap<Var<'ast>, &'ast Node<'ast>>,
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
                    map.insert(Var { ident: &unary.output }, node);
                },
                Node::Binary(binary, ..) => {
                    map.insert(Var { ident: &binary.output }, node);
                },
                Node::Input(input) => {
                    map.insert(Var { ident: &input.ident }, node);
                },
            }
        }
        Self { map }
    }

    pub(crate) fn to_graph2(&self) -> Graph2<'ast> {
        let mut map = HashMap::new();
        for (k, v) in self.map.iter() {
            map.insert(Var2 { ident: k.ident.clone() }, *v);
        }
        Graph2 { map }
    }
}

#[derive(Clone)]
pub(crate) struct Graph2<'ast> {
    pub(crate) map: HashMap<Var2, &'ast Node<'ast>>,
}