use petgraph::graph::NodeIndex;

#[derive(Clone)]
pub(crate) struct PhiFunction {
    pub(crate) name: syn::Ident,
    pub(crate) args: Vec<syn::Ident>,
    pub(crate) preds: Vec<NodeIndex>,
    pub(crate) origin_var: syn::Ident,
}

impl PhiFunction {
    pub(crate) fn new(name: syn::Ident, args: Vec<syn::Ident>, preds: Vec<NodeIndex>) -> Self {
        let origin_var = name.clone();
        Self { name, args, origin_var, preds }
    }
}

impl std::fmt::Debug for PhiFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "let {} = phi({})",
            self.name,
            self.args
                .iter()
                .zip(self.preds.iter())
                .map(|(arg, pred)| format!("{}: {}", arg, pred.index()))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
