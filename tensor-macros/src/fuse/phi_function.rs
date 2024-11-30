use petgraph::graph::NodeIndex;

#[derive(Clone)]
pub(crate) struct PhiFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) preds: Vec<NodeIndex>,
    pub(crate) origin_var: String,
}

impl PhiFunction {
    pub(crate) fn new(name: String, args: Vec<String>, preds: Vec<NodeIndex>) -> Self {
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
