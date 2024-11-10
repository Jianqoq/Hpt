use std::collections::HashMap;

pub(crate) struct SSAContext {
    // 记录每个基础变量名当前的版本号
    counters: HashMap<String, usize>,
    // 记录变量的最新 SSA 名称
    current_names: HashMap<String, String>,
}

impl SSAContext {
    pub(crate) fn new() -> Self {
        Self {
            counters: HashMap::new(),
            current_names: HashMap::new(),
        }
    }

    pub(crate) fn fresh_name(&mut self, base_name: &str) -> String {
        let counter = self.counters
            .entry(base_name.to_string())
            .and_modify(|c| {
                *c += 1;
            })
            .or_insert(1);
        let ssa_name = format!("{}_{}", base_name, counter);
        self.current_names.insert(base_name.to_string(), ssa_name.clone());
        ssa_name
    }

    pub(crate) fn fresh_expr(&mut self, expr: &syn::Expr) {
        if let syn::Expr::Path(path) = expr {
            if let Some(ident) = path.path.get_ident() {
                let string = ident.to_string();
                let counter = self.counters
                    .entry(string.clone())
                    .and_modify(|c| {
                        *c += 1;
                    })
                    .or_insert(1);
                let ssa_name = format!("{}_{}", string, counter);
                self.current_names.insert(string, ssa_name.clone());
            }
        }
    }

    pub(crate) fn current_name(&self, base_name: &str) -> Option<&String> {
        self.current_names.get(base_name)
    }

    pub(crate) fn current_name_expr(&self, expr: &syn::Expr) -> Option<&String> {
        if let syn::Expr::Path(path) = expr {
            if let Some(ident) = path.path.get_ident() {
                self.current_name(&ident.to_string())
            } else {
                None
            }
        } else {
            None
        }
    }
}
