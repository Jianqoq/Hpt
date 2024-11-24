#[derive(Clone)]
pub(crate) struct PhiFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) origin_var: String,
}

impl PhiFunction {
    pub(crate) fn new(name: String, args: Vec<String>) -> Self {
        let origin_var = name.clone();
        Self { name, args, origin_var }
    }
}

impl std::fmt::Debug for PhiFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "let {} = phi({})", self.name, self.args.join(", "))
    }
}
