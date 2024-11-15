use std::collections::{ HashMap, HashSet };

use petgraph::algo::dominators::Dominators;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use quote::ToTokens;
use syn::visit::Visit;
use syn::{ parse_quote, Stmt };

#[derive(Clone)]
pub(crate) struct CustomStmt {
    stmt: syn::Stmt,
}

impl std::fmt::Debug for CustomStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stmt.to_token_stream().to_string())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BasicBlock {
    pub(crate) statements: Vec<CustomStmt>,
    pub(crate) stmt_defs: Vec<String>,
    pub(crate) origin_vars: HashSet<String>,
    phi_functions: Vec<PhiFunction>, // 用于 SSA（后续步骤）
}

// Phi 函数结构（用于 SSA）
#[derive(Clone)]
struct PhiFunction {
    variable: String,
    origin_var: String,
    arguments: Vec<String>,
    origin_args: Vec<String>,
}

impl std::fmt::Debug for PhiFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = phi({})", self.variable, self.arguments.join(", "))
    }
}

// CFG 结构
pub(crate) struct CFG {
    pub(crate) graph: Graph<BasicBlock, ()>,
    pub(crate) entry: NodeIndex,
    pub(crate) live_in: HashMap<NodeIndex, HashSet<String>>,
    pub(crate) live_out: HashMap<NodeIndex, HashSet<String>>,
}

impl CFG {
    pub(crate) fn new() -> Self {
        let mut graph = Graph::<BasicBlock, ()>::new();
        let entry_block = BasicBlock {
            statements: vec![],
            phi_functions: vec![],
            origin_vars: HashSet::new(),
            stmt_defs: vec![],
        };
        let entry = graph.add_node(entry_block);
        CFG { graph, entry, live_in: HashMap::new(), live_out: HashMap::new() }
    }

    fn add_block(&mut self, block: BasicBlock) -> NodeIndex {
        self.graph.add_node(block)
    }

    pub(crate) fn connect(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, ());
    }

    pub(crate) fn get_variable_definitions(&self) -> HashMap<String, Vec<NodeIndex>> {
        let mut definitions: HashMap<String, Vec<NodeIndex>> = HashMap::new();

        for node in self.graph.node_indices() {
            for stmt in &self.graph[node].statements {
                if let CustomStmt { stmt: syn::Stmt::Local(local) } = stmt {
                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                        let var = pat_ident.ident.to_string();
                        definitions.entry(var).or_insert_with(Vec::new).push(node);
                    }
                }
            }
        }

        definitions
    }

    pub(crate) fn insert_phi_functions(
        &mut self,
        dominance_frontiers: &HashMap<NodeIndex, HashSet<NodeIndex>>,
        variable_definitions: &HashMap<String, Vec<NodeIndex>>
    ) {
        let mut has_already: HashMap<String, HashSet<NodeIndex>> = HashMap::new();

        for var in variable_definitions.keys() {
            has_already.insert(var.clone(), HashSet::new());
        }

        for (var, defs) in variable_definitions {
            let mut work = defs.clone();
            while let Some(def) = work.pop() {
                if let Some(frontiers) = dominance_frontiers.get(&def) {
                    for frontier in frontiers {
                        if !has_already[var].contains(frontier) {
                            // 插入 Φ 函数
                            let phi_ident = syn::Ident::new(
                                &format!("{}", var),
                                proc_macro2::Span::call_site()
                            );
                            let phi_stmt: syn::Stmt =
                                parse_quote! {
                                    let #phi_ident = __phi();
                                };
                            self.graph
                                .node_weight_mut(*frontier)
                                .unwrap()
                                .statements.insert(0, CustomStmt { stmt: phi_stmt });
                            let preds_cnt = self.graph
                                .neighbors_directed(*frontier, petgraph::Direction::Incoming)
                                .count();
                            self.graph
                                .node_weight_mut(*frontier)
                                .unwrap()
                                .phi_functions.push(PhiFunction {
                                    variable: var.clone(),
                                    arguments: vec![var.clone(); preds_cnt],
                                    origin_args: vec![var.clone(); preds_cnt],
                                    origin_var: var.clone(),
                                });
                            has_already.get_mut(var).unwrap().insert(*frontier);

                            // **只有当变量 v 不在 frontier 的 orig 中时，才将 frontier 加入工作集**
                            if !self.graph[*frontier].origin_vars.contains(var) {
                                work.push(*frontier);
                            }
                        }
                    }
                }
            }
        }
    }
}

// 变量重命名
pub(crate) fn rename_variables(cfg: &mut CFG, dominators: &Dominators<NodeIndex>) {
    let mut stacks: HashMap<String, Vec<usize>> = HashMap::new();
    let mut versions: HashMap<String, usize> = HashMap::new();

    // 收集所有变量
    let variables: HashSet<String> = cfg.graph
        .node_indices()
        .flat_map(|node| {
            cfg.graph[node].statements
                .iter()
                .filter_map(|stmt| {
                    if let CustomStmt { stmt: syn::Stmt::Local(local) } = stmt {
                        if let syn::Pat::Ident(pat_ident) = &local.pat {
                            Some(pat_ident.ident.to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for node in cfg.graph.node_indices() {
        if let Some(block) = cfg.graph.node_weight_mut(node) {
            for stmt in &mut block.statements {
                match stmt {
                    CustomStmt { stmt: syn::Stmt::Local(local) } => {
                        if let syn::Pat::Ident(pat_ident) = &mut local.pat {
                            let var = pat_ident.ident.to_string();
                            block.origin_vars.insert(var.clone());
                            block.stmt_defs.push(var);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // 初始化栈和版本
    for var in &variables {
        stacks.insert(var.clone(), Vec::new());
        versions.insert(var.clone(), 0);
    }

    rename(cfg, cfg.entry, &mut stacks, &mut versions, dominators);
}

fn rename(
    cfg: &mut CFG,
    node: NodeIndex,
    stacks: &mut HashMap<String, Vec<usize>>,
    versions: &mut HashMap<String, usize>,
    dominators: &Dominators<NodeIndex>
) {
    for phi in &mut cfg.graph[node].phi_functions {
        let var = &phi.origin_var;
        phi.variable = new_name(var, versions, stacks);
    }
    for stmt in &mut cfg.graph[node].statements {
        match &mut stmt.stmt {
            Stmt::Local(local) => {
                if let Some(init) = &mut local.init {
                    replace_vars(&mut init.expr, stacks);
                }
                if let syn::Pat::Ident(pat_ident) = &mut local.pat {
                    let new_name = new_name(&pat_ident.ident.to_string(), versions, stacks);
                    pat_ident.ident = syn::Ident::new(&new_name, pat_ident.ident.span());
                }
            }
            Stmt::Item(_) => unimplemented!("rename::Stmt::Item"),
            Stmt::Expr(expr, ..) => {
                replace_vars(expr, stacks);
            }
            Stmt::Macro(_) => unimplemented!("rename::Stmt::Macro"),
        }
    }
    for phi in &mut cfg.graph[node].phi_functions {
        let var = &mut phi.origin_var;
        if let Some(stack) = stacks.get_mut(var) {
            if let Some(&current_version) = stack.last() {
                phi.variable = format!("{}{}", var, current_version);
            }
        }
    }
    let mut succs: Vec<NodeIndex> = cfg.graph
        .neighbors_directed(node, petgraph::Direction::Outgoing)
        .collect();
    // for each succ of current node in the cfg
    //  fill in phi function parameters
    succs.sort();
    for (j, succ) in succs.iter().enumerate() {
        for phi in &mut cfg.graph[*succ].phi_functions {
            let var = &phi.origin_var;
            if let Some(stack) = stacks.get(var) {
                if let Some(&current_version) = stack.last() {
                    phi.arguments[j] = format!("{}{}", var, current_version);
                }
            }
        }
    }
    let mut dom_succs = Vec::new();
    for node_idx in cfg.graph.node_indices() {
        if dominators.immediate_dominator(node_idx) == Some(node) {
            dom_succs.push(node_idx);
        }
    }

    for succ in dom_succs {
        rename(cfg, succ, stacks, versions, dominators);
    }
    for phi in &mut cfg.graph[node].phi_functions {
        let var = &mut phi.origin_var;
        stacks.get_mut(var).expect(&format!("rename::phi::stacks: {}", var)).pop();
    }
    for stmt in &mut cfg.graph[node].stmt_defs {
        stacks.get_mut(stmt).expect(&format!("rename::stmt_defs::stacks: {}", stmt)).pop();
    }
}

fn replace_vars(expr: &mut syn::Expr, stacks: &HashMap<String, Vec<usize>>) {
    struct VarRenamer<'a> {
        stacks: &'a HashMap<String, Vec<usize>>,
    }
    impl<'a> syn::visit_mut::VisitMut for VarRenamer<'a> {
        fn visit_expr_mut(&mut self, node: &mut syn::Expr) {
            match node {
                syn::Expr::Path(expr_path) => {
                    if expr_path.qself.is_none() && expr_path.path.segments.len() == 1 {
                        let var = expr_path.path.segments[0].ident.to_string();
                        if let Some(stack) = self.stacks.get(&var) {
                            if let Some(current_cnt) = stack.last() {
                                expr_path.path.segments[0].ident = syn::Ident::new(
                                    &format!("{}{}", var, current_cnt),
                                    expr_path.path.segments[0].ident.span()
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
            syn::visit_mut::visit_expr_mut(self, node);
        }
    }
    syn::visit_mut::visit_expr_mut(&mut (VarRenamer { stacks }), expr);
}

fn new_name(
    var: &str,
    versions: &mut HashMap<String, usize>,
    stacks: &mut HashMap<String, Vec<usize>>
) -> String {
    if let Some(cnt) = versions.get_mut(var) {
        *cnt += 1;
        if let Some(stack) = stacks.get_mut(var) {
            stack.push(*cnt);
        }
        format!("{}{}", var, cnt)
    } else {
        panic!("{} not found in new_name", var);
    }
}

pub(crate) struct CFGBuilder<'a> {
    pub(crate) cfg: &'a mut CFG,
    pub(crate) current_block: NodeIndex,
    loop_stack: Vec<NodeIndex>,
}

impl<'a> CFGBuilder<'a> {
    pub(crate) fn new(cfg: &'a mut CFG) -> Self {
        let entry = cfg.entry.clone();
        CFGBuilder {
            cfg,
            current_block: entry,
            loop_stack: Vec::new(),
        }
    }

    // 创建新的基本块并返回其索引
    fn new_block(&mut self) -> NodeIndex {
        let block = BasicBlock {
            statements: vec![],
            phi_functions: vec![],
            origin_vars: HashSet::new(),
            stmt_defs: vec![],
        };
        self.cfg.add_block(block)
    }

    // 连接当前块到目标块
    fn connect_to(&mut self, to: NodeIndex) {
        self.cfg.connect(self.current_block, to);
    }

    fn set_current_block(&mut self, new_block: NodeIndex) {
        self.current_block = new_block;
    }

    // 处理 if 语句
    fn handle_if(&mut self, expr_if: &syn::ExprIf) {
        // 创建 then 分支块
        let then_block = self.new_block();
        // 创建 else 分支块
        let else_block = self.new_block();
        // 创建合并块
        let merge_block = self.new_block();

        // 连接当前块到 then 和 else 分支
        self.connect_to(then_block);
        self.connect_to(else_block);

        // 处理 then 分支
        self.set_current_block(then_block);
        self.visit_block(&expr_if.then_branch);
        self.connect_to(merge_block);

        // 处理 else 分支
        self.set_current_block(else_block);
        if let Some(else_branch) = &expr_if.else_branch {
            match &else_branch.1.as_ref() {
                syn::Expr::Block(expr_block) => {
                    self.visit_block(&expr_block.block);
                }
                _ => {
                    // 其他表达式类型
                    self.visit_expr(&else_branch.1);
                }
            }
        }
        self.connect_to(merge_block);

        // 更新当前块为合并块
        self.set_current_block(merge_block);
    }

    // 处理 loop 语句
    fn handle_loop(&mut self, expr_loop: &syn::ExprLoop) {
        // 创建循环体块
        let loop_block = self.new_block();
        // 创建循环后的块
        let after_loop_block = self.new_block();

        // 连接当前块到循环体块
        self.connect_to(loop_block);

        // 将循环后的块推入栈中，以便处理 break 语句
        self.loop_stack.push(after_loop_block);

        // 设置当前块为循环体块，并处理循环体
        self.set_current_block(loop_block);
        self.visit_block(&expr_loop.body);

        // 连接循环体块回到自身，表示下一次迭代
        self.connect_to(loop_block);
        // 连接循环体块到循环后的块，表示退出循环
        self.connect_to(after_loop_block);

        // 处理完循环后，将循环后的块从栈中弹出
        self.loop_stack.pop();

        // 更新当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
    }

    fn handle_break(&mut self, _expr_break: &syn::ExprBreak) {
        if let Some(after_loop_block) = self.loop_stack.last() {
            self.connect_to(*after_loop_block);
        } else {
            // 处理非循环中的 break，可以选择报错或其他处理方式
            eprintln!("Error: 'break' outside of loop");
        }
        // 创建一个新的块，因为 break 语句通常会终止当前块
        let new_block = self.new_block();
        self.set_current_block(new_block);
    }

    // 处理 continue 语句
    fn handle_continue(&mut self, _expr_continue: &syn::ExprContinue) {
        if let Some(after_loop_block) = self.loop_stack.last() {
            // 假设循环体块是上一个块，重新连接到循环体块以进行下一次迭代
            // 你可能需要更精确地跟踪循环入口块
            // 这里简化处理，连接回当前循环体块
            // 你需要确保 `loop_block` 是循环体块的 NodeIndex
            // 这里假设 `loop_block` 是当前循环体块
            if let Some(&loop_block) = self.loop_stack.iter().rev().nth(1) {
                self.connect_to(loop_block);
            } else {
                // 如果无法找到循环体块，可以选择报错或其他处理方式
                eprintln!("Error: 'continue' without loop body reference");
            }
        } else {
            // 处理非循环中的 continue，可以选择报错或其他处理方式
            eprintln!("Error: 'continue' outside of loop");
        }
        // 创建一个新的块，因为 continue 语句通常会终止当前块
        let new_block = self.new_block();
        self.set_current_block(new_block);
    }

    // 处理 match 语句
    fn handle_match(&mut self, expr_match: &syn::ExprMatch) {
        // 创建合并块
        let merge_block = self.new_block();

        for arm in &expr_match.arms {
            // 为每个匹配分支创建一个块
            let arm_block = self.new_block();
            // 连接当前块到分支块
            self.connect_to(arm_block);
            // 处理分支块中的语句
            self.set_current_block(arm_block);
            self.visit_expr(&arm.body);
            // 连接分支块到合并块
            self.connect_to(merge_block);
        }

        // 更新当前块为合并块
        self.set_current_block(merge_block);
    }

    fn handle_while_loop(&mut self, expr_while: &syn::ExprWhile) {
        // 创建条件检查块
        let condition_block = self.new_block();
        // 创建循环体块
        let loop_block = self.new_block();
        // 创建循环后的块
        let after_loop_block = self.new_block();

        // 连接当前块到条件检查块
        self.connect_to(condition_block);

        // 连接条件检查块到循环体块（如果条件为真）和循环后的块（如果条件为假）
        self.set_current_block(condition_block);
        if let Some(block) = self.cfg.graph.node_weight_mut(condition_block) {
            // 你可以将条件表达式添加到条件检查块中
            block.statements.push(CustomStmt { stmt: Stmt::Expr(*expr_while.cond.clone(), None) });
        }

        // 创建两个新的连接：条件为真和条件为假
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // Push after_loop_block to loop_stack for handling break
        self.loop_stack.push(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.visit_block(&expr_while.body);

        // 连接循环体块回到条件检查块（表示下一次迭代）
        self.connect_to(condition_block);
        // 连接循环体块到循环后的块（如果有 break）
        self.connect_to(after_loop_block);

        // Pop after_loop_block from loop_stack
        self.loop_stack.pop();

        // 设置当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
    }

    fn handle_for_loop(&mut self, expr_for: &syn::ExprForLoop) {
        // 创建迭代器初始化块
        let init_block = self.new_block();
        // 创建条件检查块
        let condition_block = self.new_block();
        // 创建循环体块
        let loop_block = self.new_block();
        // 创建循环后的块
        let after_loop_block = self.new_block();

        // 连接当前块到初始化块
        self.connect_to(init_block);

        // 处理迭代器初始化和元素绑定
        self.set_current_block(init_block);
        // 创建 `let pat = expr` 语句
        let pat = &expr_for.pat;
        let local = quote::quote! {
            #[for_loop_var]
            let #pat;
        };
        let stmt: Stmt = syn::parse2(local).expect("cfg::handle_for_loop::local");
        if let Some(block) = self.cfg.graph.node_weight_mut(init_block) {
            block.statements.push(CustomStmt { stmt });
        }

        // 连接初始化块到条件检查块
        self.connect_to(condition_block);

        // 处理条件检查
        self.set_current_block(condition_block);
        if let Some(block) = self.cfg.graph.node_weight_mut(condition_block) {
            let expr = &expr_for.expr;
            block.statements.push(CustomStmt { stmt: Stmt::Expr(*(*expr).clone(), None) });
        }

        // 创建连接：条件为真进入循环体块，条件为假进入循环后的块
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // Push after_loop_block to loop_stack for handling break
        self.loop_stack.push(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.visit_block(&expr_for.body);

        // 连接循环体块回到条件检查块（表示下一次迭代）
        self.connect_to(condition_block);
        // 连接循环体块到循环后的块（如果有 break）
        self.connect_to(after_loop_block);

        // Pop after_loop_block from loop_stack
        self.loop_stack.pop();

        // 设置当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
    }
}

impl<'ast, 'a> Visit<'ast> for CFGBuilder<'a> {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let new_block = self.new_block();
        for arg in &i.sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => {}
                syn::FnArg::Typed(pat_type) => {
                    match &*pat_type.pat {
                        syn::Pat::Ident(pat_ident) => {
                            if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                                let local = syn
                                    ::parse2(
                                        quote::quote! {
                                    let #pat_ident;
                                }
                                    )
                                    .unwrap();
                                block.statements.push(CustomStmt { stmt: local });
                            }
                        }
                        _ =>
                            unimplemented!(
                                "cfg_builder::visit_item_fn::fn_arg::typed::{}",
                                pat_type.to_token_stream().to_string()
                            ),
                    }
                }
            }
        }
        self.connect_to(new_block);
        self.set_current_block(new_block);
        self.visit_block(&i.block);
    }

    fn visit_expr_break(&mut self, expr_break: &'ast syn::ExprBreak) {
        self.handle_break(expr_break);
    }

    fn visit_expr_continue(&mut self, expr_continue: &'ast syn::ExprContinue) {
        self.handle_continue(expr_continue);
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        self.handle_if(i);
    }

    fn visit_expr_loop(&mut self, i: &'ast syn::ExprLoop) {
        self.handle_loop(i);
    }

    fn visit_expr_match(&mut self, i: &'ast syn::ExprMatch) {
        self.handle_match(i);
    }

    fn visit_expr_for_loop(&mut self, i: &'ast syn::ExprForLoop) {
        self.handle_for_loop(i);
    }

    fn visit_expr(&mut self, node: &'ast syn::Expr) {
        match node {
            syn::Expr::Block(_binding_0) => {
                self.visit_expr_block(_binding_0);
            }
            syn::Expr::ForLoop(_binding_0) => {
                self.visit_expr_for_loop(_binding_0);
            }
            syn::Expr::If(_binding_0) => {
                self.visit_expr_if(_binding_0);
            }
            syn::Expr::Loop(_binding_0) => {
                self.visit_expr_loop(_binding_0);
            }
            syn::Expr::Match(_binding_0) => {
                self.visit_expr_match(_binding_0);
            }
            syn::Expr::While(_binding_0) => {
                self.visit_expr_while(_binding_0);
            }
            _ => {
                if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                    block.statements.push(CustomStmt { stmt: Stmt::Expr(node.clone(), None) });
                }
            }
        }
    }

    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        match node {
            Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    let mut has_control_flow = false;
                    match &init.expr.as_ref() {
                        | syn::Expr::ForLoop(_)
                        | syn::Expr::If(_)
                        | syn::Expr::Loop(_)
                        | syn::Expr::Match(_)
                        | syn::Expr::While(_) => {
                            has_control_flow = true;
                        }
                        _ => {}
                    }
                    if has_control_flow {
                        syn::visit::visit_stmt(self, node);
                    } else {
                        if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                            block.statements.push(CustomStmt { stmt: node.clone() });
                        }
                    }
                }
            }
            Stmt::Item(_) => {
                if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                    block.statements.push(CustomStmt { stmt: node.clone() });
                }
            }
            Stmt::Expr(expr, _) => {
                let mut has_control_flow = false;
                match &expr {
                    | syn::Expr::ForLoop(_)
                    | syn::Expr::If(_)
                    | syn::Expr::Loop(_)
                    | syn::Expr::Match(_)
                    | syn::Expr::While(_) => {
                        has_control_flow = true;
                    }
                    _ => {}
                }
                if has_control_flow {
                    syn::visit::visit_stmt(self, node);
                } else {
                    if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                        block.statements.push(CustomStmt { stmt: node.clone() });
                    }
                }
            }
            Stmt::Macro(_) => unimplemented!("cfg::visit_stmt::Macro"),
        }
    }
}
