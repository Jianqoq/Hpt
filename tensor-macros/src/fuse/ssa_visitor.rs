use syn::{ visit_mut::*, Expr, ExprAssign, ExprLet, Ident, ItemFn, Stmt, Block, ExprIf, Pat };
use quote::{ format_ident, ToTokens };
use std::collections::HashMap;
use std::collections::HashSet;

/// 结构体用于跟踪变量的当前版本
pub(crate) struct SSATransformer {
    // 变量名到当前版本的映射
    var_versions: HashMap<String, usize>,
    // 变量名到版本栈的映射，用于恢复变量版本
    var_stacks: HashMap<String, Vec<usize>>,
    current_block: Vec<*mut syn::Block>,
    assigned_vars: HashSet<String>,
}

impl SSATransformer {
    pub(crate) fn new() -> Self {
        SSATransformer {
            var_versions: HashMap::new(),
            var_stacks: HashMap::new(),
            current_block: Vec::new(),
            assigned_vars: HashSet::new(),
        }
    }

    /// 获取变量的当前版本
    fn get_current_version(&self, var: &str) -> Option<usize> {
        self.var_versions.get(var).copied()
    }

    /// 新增一个变量版本
    fn add_version(&mut self, var: &str) -> usize {
        let counter = self.var_versions.entry(var.to_string()).or_insert(0);
        *counter += 1;
        let version = *counter;
        self.var_stacks.entry(var.to_string()).or_default().push(version);
        version
    }

    /// 恢复变量的上一个版本
    fn pop_version(&mut self, var: &str) {
        if let Some(stack) = self.var_stacks.get_mut(var) {
            stack.pop();
            if let Some(&last_version) = stack.last() {
                self.var_versions.insert(var.to_string(), last_version);
            } else {
                self.var_versions.remove(var);
            }
        }
    }

    /// 获取变量名带版本后缀
    fn get_var_with_version(&self, var: &str) -> String {
        if let Some(version) = self.get_current_version(var) {
            format!("{}_{}", var, version)
        } else {
            var.to_string()
        }
    }

    /// 重命名变量为带版本的变量
    fn rename_ident(&self, ident: &Ident) -> Ident {
        let var_name = ident.to_string();
        if let Some(version) = self.get_current_version(&var_name) {
            format_ident!("{}_{}", var_name, version)
        } else {
            ident.clone()
        }
    }
}

impl VisitMut for SSATransformer {
    fn visit_item_fn_mut(&mut self, node: &mut ItemFn) {
        self.var_versions.clear();
        self.var_stacks.clear();
        visit_item_fn_mut(self, node);
    }

    fn visit_pat_mut(&mut self, i: &mut syn::Pat) {
        if let Pat::Ident(pat_ident) = i {
            let var_name = pat_ident.ident.to_string();
            println!("var_name: {}", var_name);
            let version = self.add_version(&var_name);
            let new_ident = format_ident!("{}_{}", var_name, version);
            pat_ident.ident = new_ident;
        }
    }

    fn visit_fn_arg_mut(&mut self, arg: &mut syn::FnArg) {
        match arg {
            syn::FnArg::Receiver(_binding_0) => {
                self.visit_receiver_mut(_binding_0);
            }
            syn::FnArg::Typed(_binding_0) => {
                self.visit_pat_type_mut(_binding_0);
            }
        }
    }

    fn visit_expr_path_mut(&mut self, node: &mut syn::ExprPath) {
        if let Some(ident) = node.path.get_ident() {
            let var_name = ident.to_string();
            if let Some(version) = self.get_current_version(&var_name) {
                let new_ident = format_ident!("{}_{}", var_name, version);
                node.path = syn::Path::from(new_ident);
            }
        }
    }

    fn visit_expr_assign_mut(&mut self, node: &mut syn::ExprAssign) {
        self.visit_attributes_mut(&mut node.attrs);
        if let Expr::Path(ref path) = *node.left {
            if let Some(ident) = path.path.get_ident() {
                let var_name = ident.to_string();
                self.assigned_vars.insert(var_name.clone());
            }
        }
        self.visit_expr_mut(&mut *node.left);
        self.visit_expr_mut(&mut *node.right);
    }

    fn visit_expr_if_mut(&mut self, node: &mut syn::ExprIf) {
        self.visit_attributes_mut(&mut node.attrs);
        // 处理条件表达式
        self.visit_expr_mut(&mut *node.cond);
        // 保存当前变量版本
        let saved_var_versions = self.var_versions.clone();
        let saved_var_stacks = self.var_stacks.clone();
        // 处理 'then' 分支
        self.visit_block_mut(&mut node.then_branch);
        // // 保存 'then' 分支后的变量版本
        let then_var_versions = self.var_versions.clone();
        let then_var_stacks = self.var_stacks.clone();
        // // 恢复到 if 前的变量版本
        self.var_versions = saved_var_versions.clone();
        self.var_stacks = saved_var_stacks.clone();
        // 处理 'else' 分支
        if let Some((_, else_branch)) = &mut node.else_branch {
            match else_branch.as_mut() {
                Expr::Block(ref mut else_block) => {
                    self.visit_block_mut(&mut else_block.block);
                }
                Expr::If(ref mut else_if) => {
                    self.visit_expr_if_mut(else_if);
                }
                _ => {
                    self.visit_expr_mut(else_branch);
                }
            }
        }
        // 保存 'else' 分支后的变量版本
        let else_var_versions = self.var_versions.clone();
        let else_var_stacks = self.var_stacks.clone();

        // 计算合并后的变量版本
        let mut merged_vars = HashMap::new();
        for var in then_var_versions.keys().chain(else_var_versions.keys()) {
            let then_version = then_var_versions.get(var);
            let else_version = else_var_versions.get(var);
            if then_version != else_version {
                // 变量在分支中被赋值，需插入 Phi 函数
                merged_vars.insert(var.clone(), (then_version.cloned(), else_version.cloned()));
            }
        }
        // 生成 Phi 函数绑定
        if let Some(current_block) = self.current_block.last_mut() {
            let mut phi_stmts = Vec::new();
            for (var, (then_ver, else_ver)) in merged_vars {
                let new_version = self.add_version(&var);
                let new_ident = format_ident!("{}_{}", var, new_version);
                let cond = &node.cond;
                let then_ident = format_ident!("{}_{}", var, then_ver.unwrap_or(0));
                let else_ident = format_ident!("{}_{}", var, else_ver.unwrap_or(0));
                let new_bind =
                    quote::quote!(
                    let #new_ident = if #cond {
                        #then_ident
                    } else {
                        #else_ident
                    };
                );
                let phi_stmt: Stmt = syn::parse2(new_bind).expect("Failed to parse phi_bind");

                phi_stmts.push(phi_stmt);
            }
        }

        // // 恢复变量版本到 if 分支后的状态
        // self.var_versions = saved_var_versions;
        // self.var_stacks = saved_var_stacks;
    }

    fn visit_block_mut(&mut self, node: &mut syn::Block) {
        self.current_block.push(node);
        visit_block_mut(self, node);
        self.current_block.pop();
    }
}
