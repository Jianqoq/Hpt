use std::collections::{ HashMap, HashSet };

use petgraph::algo::dominators::Dominators;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use quote::ToTokens;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::Stmt;

use super::codegen;
use super::phi_function::PhiFunction;
use super::ty_infer::Type;
use super::use_define_visitor::UseDefineVisitor;
use super::var_recover::VarRecover;

#[derive(Clone)]
pub(crate) struct CustomStmt {
    pub(crate) stmt: syn::Stmt,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub(crate) enum BlockType {
    Normal,
    IfCond,
    IfThen,
    IfElse,
    ForInit,
    ForBody,
    ForCond,
    WhileCond,
    WhileBody,
    LoopBody,
}

impl std::fmt::Debug for CustomStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stmt.to_token_stream().to_string())
    }
}

impl ToTokens for CustomStmt {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.stmt.to_tokens(tokens);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BasicBlock {
    pub(crate) statements: Vec<CustomStmt>,
    pub(crate) phi_functions: Vec<PhiFunction>,
    pub(crate) origin_vars: HashSet<String>,
    pub(crate) defined_vars: HashSet<String>,
    pub(crate) assigned_vars: HashSet<String>,
    pub(crate) used_vars: HashSet<String>,
    pub(crate) block_type: BlockType,
    pub(crate) live_in: HashSet<String>,
    pub(crate) live_out: HashSet<String>,
    pub(crate) origin_var_map: HashMap<String, String>,
}

// CFG 结构
pub(crate) struct CFG {
    pub(crate) graph: Graph<BasicBlock, ()>,
    pub(crate) entry: NodeIndex,
}

fn visit_pat(pat: &syn::Pat, definitions: &mut HashMap<String, Vec<NodeIndex>>, node: NodeIndex) {
    match pat {
        syn::Pat::Const(_) => unimplemented!("visit_pat::Pat::Const"),
        syn::Pat::Ident(pat_ident) => {
            let var = pat_ident.ident.to_string();
            definitions.entry(var).or_insert_with(Vec::new).push(node);
        }
        syn::Pat::Lit(_) => unimplemented!("visit_pat::Pat::Lit"),
        syn::Pat::Macro(_) => unimplemented!("visit_pat::Pat::Macro"),
        syn::Pat::Or(_) => unimplemented!("visit_pat::Pat::Or"),
        syn::Pat::Paren(_) => unimplemented!("visit_pat::Pat::Paren"),
        syn::Pat::Path(_) => unimplemented!("visit_pat::Pat::Path"),
        syn::Pat::Range(_) => unimplemented!("visit_pat::Pat::Range"),
        syn::Pat::Reference(_) => unimplemented!("visit_pat::Pat::Reference"),
        syn::Pat::Rest(_) => unimplemented!("visit_pat::Pat::Rest"),
        syn::Pat::Slice(_) => unimplemented!("visit_pat::Pat::Slice"),
        syn::Pat::Struct(_) => unimplemented!("visit_pat::Pat::Struct"),
        syn::Pat::Tuple(_) => unimplemented!("visit_pat::Pat::Tuple"),
        syn::Pat::TupleStruct(_) => unimplemented!("visit_pat::Pat::TupleStruct"),
        syn::Pat::Type(ty) => {
            visit_pat(&ty.pat, definitions, node);
        }
        syn::Pat::Verbatim(_) => unimplemented!("visit_pat::Pat::Verbatim"),
        syn::Pat::Wild(_) => {}
        _ => unimplemented!("visit_pat::Pat::Other"),
    }
}

fn visit_expr_assign_add_var(
    expr: &syn::ExprAssign,
    definitions: &mut HashMap<String, Vec<NodeIndex>>,
    node: NodeIndex
) {
    match expr.left.as_ref() {
        syn::Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                definitions.entry(ident.to_string()).or_insert_with(Vec::new).push(node);
            }
        }
        _ => unimplemented!("visit_expr_assign_add_var::Expr::Other"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum VarStatus {
    Assigned,
    Redeclare,
    NotFound,
}

fn check_expr_assign(expr: &syn::ExprAssign, var: &str, status: &mut VarStatus) {
    match &expr.left.as_ref() {
        syn::Expr::Array(_) => unimplemented!("cfg::insert_phi_functions::Expr::Array"),
        syn::Expr::Assign(_) => unimplemented!("cfg::insert_phi_functions::Expr::Assign"),
        syn::Expr::Async(_) => unimplemented!("cfg::insert_phi_functions::Expr::Async"),
        syn::Expr::Await(_) => unimplemented!("cfg::insert_phi_functions::Expr::Await"),
        syn::Expr::Binary(_) => unimplemented!("cfg::insert_phi_functions::Expr::Binary"),
        syn::Expr::Block(_) => unimplemented!("cfg::insert_phi_functions::Expr::Block"),
        syn::Expr::Break(_) => unimplemented!("cfg::insert_phi_functions::Expr::Break"),
        syn::Expr::Call(_) => unimplemented!("cfg::insert_phi_functions::Expr::Call"),
        syn::Expr::Cast(_) => unimplemented!("cfg::insert_phi_functions::Expr::Cast"),
        syn::Expr::Closure(_) => unimplemented!("cfg::insert_phi_functions::Expr::Closure"),
        syn::Expr::Const(_) => unimplemented!("cfg::insert_phi_functions::Expr::Const"),
        syn::Expr::Continue(_) => unimplemented!("cfg::insert_phi_functions::Expr::Continue"),
        syn::Expr::Field(_) => unimplemented!("cfg::insert_phi_functions::Expr::Field"),
        syn::Expr::ForLoop(_) => unimplemented!("cfg::insert_phi_functions::Expr::ForLoop"),
        syn::Expr::Group(_) => unimplemented!("cfg::insert_phi_functions::Expr::Group"),
        syn::Expr::If(_) => unimplemented!("cfg::insert_phi_functions::Expr::If"),
        syn::Expr::Index(_) => unimplemented!("cfg::insert_phi_functions::Expr::Index"),
        syn::Expr::Infer(_) => unimplemented!("cfg::insert_phi_functions::Expr::Infer"),
        syn::Expr::Let(_) => unimplemented!("cfg::insert_phi_functions::Expr::Let"),
        syn::Expr::Lit(_) => unimplemented!("cfg::insert_phi_functions::Expr::Lit"),
        syn::Expr::Loop(_) => unimplemented!("cfg::insert_phi_functions::Expr::Loop"),
        syn::Expr::Macro(_) => unimplemented!("cfg::insert_phi_functions::Expr::Macro"),
        syn::Expr::Match(_) => unimplemented!("cfg::insert_phi_functions::Expr::Match"),
        syn::Expr::MethodCall(_) => unimplemented!("cfg::insert_phi_functions::Expr::MethodCall"),
        syn::Expr::Paren(_) => unimplemented!("cfg::insert_phi_functions::Expr::Paren"),
        syn::Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                if &ident.to_string() == var {
                    *status = VarStatus::Assigned;
                }
            }
        }
        syn::Expr::Range(_) => unimplemented!("cfg::insert_phi_functions::Expr::Range"),
        syn::Expr::RawAddr(_) => unimplemented!("cfg::insert_phi_functions::Expr::RawAddr"),
        syn::Expr::Reference(_) => unimplemented!("cfg::insert_phi_functions::Expr::Reference"),
        syn::Expr::Repeat(_) => unimplemented!("cfg::insert_phi_functions::Expr::Repeat"),
        syn::Expr::Return(_) => unimplemented!("cfg::insert_phi_functions::Expr::Return"),
        syn::Expr::Struct(_) => unimplemented!("cfg::insert_phi_functions::Expr::Struct"),
        syn::Expr::Try(_) => unimplemented!("cfg::insert_phi_functions::Expr::Try"),
        syn::Expr::TryBlock(_) => unimplemented!("cfg::insert_phi_functions::Expr::TryBlock"),
        syn::Expr::Tuple(_) => unimplemented!("cfg::insert_phi_functions::Expr::Tuple"),
        syn::Expr::Unary(_) => unimplemented!("cfg::insert_phi_functions::Expr::Unary"),
        syn::Expr::Unsafe(_) => unimplemented!("cfg::insert_phi_functions::Expr::Unsafe"),
        syn::Expr::Verbatim(_) => unimplemented!("cfg::insert_phi_functions::Expr::Verbatim"),
        syn::Expr::While(_) => unimplemented!("cfg::insert_phi_functions::Expr::While"),
        syn::Expr::Yield(_) => unimplemented!("cfg::insert_phi_functions::Expr::Yield"),
        _ => unimplemented!("cfg::insert_phi_functions::Expr::Other"),
    }
}

impl CFG {
    pub(crate) fn new() -> Self {
        let mut graph = Graph::<BasicBlock, ()>::new();
        let entry_block = BasicBlock {
            statements: vec![],
            origin_vars: HashSet::new(),
            defined_vars: HashSet::new(),
            used_vars: HashSet::new(),
            live_in: HashSet::new(),
            live_out: HashSet::new(),
            block_type: BlockType::Normal,
            phi_functions: vec![],
            origin_var_map: HashMap::new(),
            assigned_vars: HashSet::new(),
        };
        let entry = graph.add_node(entry_block);
        CFG { graph, entry }
    }

    pub(crate) fn live_analysis(&mut self) {
        let mut live_in = HashMap::new();
        let mut live_out = HashMap::new();

        // 初始化所有基本块的live_in和live_out集合
        for block in self.graph.node_indices() {
            live_in.insert(block, HashSet::new());
            live_out.insert(block, HashSet::new());
        }

        // 迭代计算直到不再发生变化
        let mut changed = true;
        while changed {
            changed = false;

            // 对每个基本块进行分析
            for block in self.graph.node_indices() {
                let old_in = live_in[&block].clone();
                let old_out = live_out[&block].clone();

                // 计算OUT[B] = ∪(IN[S]) 其中S是B的后继
                let mut new_out: HashSet<String> = HashSet::new();
                for succ in self.graph.neighbors_directed(block, petgraph::Direction::Outgoing) {
                    new_out.extend(live_in[&succ].iter().cloned());
                }

                // 计算IN[B] = USE[B] ∪ (OUT[B] - DEF[B])
                let block_data = &self.graph[block];
                let mut new_in = block_data.used_vars.clone();
                new_in.retain(|var| !block_data.defined_vars.contains(var));
                for var in new_out.iter() {
                    if !block_data.defined_vars.contains(var) {
                        new_in.insert(var.clone());
                    }
                }

                // 更新集合并检查是否发生变化
                *live_out.get_mut(&block).unwrap() = new_out;
                *live_in.get_mut(&block).unwrap() = new_in;

                if old_in != live_in[&block] || old_out != live_out[&block] {
                    changed = true;
                }
            }
        }

        // 将结果存储到CFG中
        for block in self.graph.node_indices() {
            self.graph[block].live_in = live_in[&block].clone();
            self.graph[block].live_out = live_out[&block].clone();
        }
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
                match &stmt.stmt {
                    Stmt::Local(local) => {
                        visit_pat(&local.pat, &mut definitions, node);
                    }
                    Stmt::Expr(expr, _) => {
                        if let syn::Expr::Assign(assign) = &expr {
                            visit_expr_assign_add_var(assign, &mut definitions, node);
                        }
                    }
                    _ => {}
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
                            let incomings = self.graph.neighbors_directed(
                                *frontier,
                                petgraph::Direction::Incoming
                            );
                            let mut indices = incomings.clone().collect::<Vec<_>>();
                            indices.sort();
                            let mut status = VarStatus::NotFound;
                            let mut has_assigned = false;
                            for incoming in incomings {
                                for stmt in &self.graph[incoming].statements {
                                    match &stmt.stmt {
                                        Stmt::Local(local) => {
                                            if let syn::Pat::Ident(pat_ident) = &local.pat {
                                                if &pat_ident.ident.to_string() == var {
                                                    status = VarStatus::Redeclare;
                                                }
                                            }
                                        }
                                        Stmt::Expr(expr, ..) => {
                                            if let syn::Expr::Assign(assign) = &expr {
                                                check_expr_assign(assign, var, &mut status);
                                            }
                                        }
                                        _ => {}
                                    }
                                    if status != VarStatus::NotFound {
                                        break;
                                    }
                                }
                                if status == VarStatus::Assigned {
                                    has_assigned = true;
                                    break;
                                } else {
                                    status = VarStatus::NotFound; // reset status
                                }
                            }
                            if has_assigned {
                                let preds_cnt = self.graph
                                    .neighbors_directed(*frontier, petgraph::Direction::Incoming)
                                    .count();
                                let args = vec![var.clone(); preds_cnt];
                                let phi_function = PhiFunction::new(var.clone(), args, indices);
                                self.graph
                                    .node_weight_mut(*frontier)
                                    .unwrap()
                                    .used_vars.insert(var.clone());
                                self.graph
                                    .node_weight_mut(*frontier)
                                    .unwrap()
                                    .phi_functions.push(phi_function);
                            }
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

    pub(crate) fn build_graphs<'ast>(
        &'ast self,
        type_table: &'ast HashMap<String, Type>
    ) -> Graph<super::build_graph::Graph<'ast>, ()> {
        let mut graph = Graph::<super::build_graph::Graph<'ast>, ()>::new();
        let mut sorted_indices = self.graph.node_indices().collect::<Vec<_>>();
        sorted_indices.sort();
        for node in sorted_indices {
            let mut comp_graph = super::build_graph::Graph::new(type_table, node.index());
            for (idx, stmt) in self.graph
                .node_weight(node)
                .expect("fuse::cfg::build_graphs::node weight not found")
                .statements.iter()
                .enumerate() {
                comp_graph.current_idx = idx;
                comp_graph.visit_stmt(&stmt.stmt);
            }
            graph.add_node(comp_graph);
        }
        for idx in self.graph.node_indices() {
            let edges = self.graph.edges(idx);
            for edge in edges {
                graph.add_edge(edge.source(), edge.target(), ());
            }
        }
        graph
    }

    // 变量重命名
    pub(crate) fn rename_variables(&mut self, dominators: &Dominators<NodeIndex>) {
        let mut stacks: HashMap<String, Vec<usize>> = HashMap::new();
        let mut versions: HashMap<String, usize> = HashMap::new();

        // 收集所有变量
        let variables: HashSet<String> = self.graph
            .node_indices()
            .flat_map(|node| {
                let vars = self.graph[node].statements
                    .iter()
                    .filter_map(|stmt| {
                        let mut visitor = UseDefineVisitor::new();
                        visitor.visit_stmt(&stmt.stmt);
                        let mut vars = HashSet::new();
                        vars.extend(visitor.used_vars.drain());
                        vars.extend(visitor.define_vars.drain());
                        vars.extend(visitor.assigned_vars.drain());
                        Some(vars)
                    })
                    .collect::<Vec<HashSet<String>>>();
                println!("node: {}", node.index());
                println!("vars: {:#?}", vars);
                for var in vars.iter().flatten() {
                    self.graph[node].origin_var_map.insert(var.clone(), var.clone());
                }
                println!("origin_var_map: {:#?}", self.graph[node].origin_var_map);
                vars.into_iter().flatten().collect::<Vec<_>>()
            })
            .collect();

        for node in self.graph.node_indices() {
            if let Some(block) = self.graph.node_weight_mut(node) {
                for stmt in &mut block.statements {
                    match stmt {
                        CustomStmt { stmt: syn::Stmt::Local(local) } => {
                            insert_origin_var(&mut block.origin_vars, &local.pat);
                        }
                        _ => {}
                    }
                }
                for phi_function in &mut block.phi_functions {
                    block.origin_vars.insert(phi_function.origin_var.clone());
                }
            }
        }

        // 初始化栈和版本
        for var in variables {
            stacks.insert(var.clone(), Vec::new());
            versions.insert(var, 0);
        }

        rename(self, self.entry, &mut stacks, &mut versions, dominators);
    }

    pub(crate) fn gen_code(&self) -> crate::TokenStream2 {
        let mut body = quote::quote!();
        let mut order = self.reverse_postorder();
        order.retain(|x| *x != self.entry.index());
        println!("order: {:#?}", order);
        for node in order {
            let block = &self.graph[NodeIndex::new(node)];
            body.extend(codegen::stmt(block));
        }
        body
    }

    // 定义函数 reverse_postorder
    fn reverse_postorder(&self) -> Vec<usize> {
        // 构建邻接表
        let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();
        for idx in self.graph.edge_indices() {
            let (src, dst) = self.graph
                .edge_endpoints(idx)
                .expect("fuse::cfg::gen_code::edge endpoints not found");
            graph.entry(src.index()).or_insert_with(Vec::new).push(dst.index());
        }

        for neighbors in graph.values_mut() {
            neighbors.sort_unstable();
            neighbors.reverse();
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut order: Vec<usize> = Vec::new();

        // 定义递归的深度优先搜索函数
        fn dfs(
            node: usize,
            graph: &HashMap<usize, Vec<usize>>,
            visited: &mut HashSet<usize>,
            order: &mut Vec<usize>
        ) {
            if visited.contains(&node) {
                return;
            }
            visited.insert(node);
            if let Some(neighbors) = graph.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        dfs(neighbor, graph, visited, order);
                    }
                }
            }
            order.push(node);
        }

        // 获取所有节点
        let mut nodes: HashSet<usize> = HashSet::new();
        for idx in self.graph.edge_indices() {
            let (src, dst) = self.graph
                .edge_endpoints(idx)
                .expect("fuse::cfg::gen_code::edge endpoints not found");
            nodes.insert(src.index());
            nodes.insert(dst.index());
        }

        // 按节点编号排序，确保遍历顺序一致
        let mut sorted_nodes: Vec<usize> = nodes.into_iter().collect();
        sorted_nodes.sort_unstable();
        // 对所有节点进行 DFS，以确保覆盖所有连通分量
        for &node in &sorted_nodes {
            if !visited.contains(&node) {
                dfs(node, &graph, &mut visited, &mut order);
            }
        }

        // 逆后序
        order.into_iter().rev().collect()
    }

    pub(crate) fn replace_all_var_back(&mut self) {
        for node in self.graph.node_indices() {
            let map = self.graph[node].origin_var_map.clone();
            for stmt in &mut self.graph[node].statements {
                let mut recover = VarRecover::new(&map);
                recover.visit_stmt_mut(&mut stmt.stmt);
            }
        }
    }
}

fn insert_origin_var(origin_vars: &mut HashSet<String>, pat: &syn::Pat) {
    match pat {
        syn::Pat::Const(_) => unimplemented!("insert_origin_var::Pat::Const"),
        syn::Pat::Ident(pat_ident) => {
            origin_vars.insert(pat_ident.ident.to_string());
        }
        syn::Pat::Lit(_) => unimplemented!("insert_origin_var::Pat::Lit"),
        syn::Pat::Macro(_) => unimplemented!("insert_origin_var::Pat::Macro"),
        syn::Pat::Or(_) => unimplemented!("insert_origin_var::Pat::Or"),
        syn::Pat::Paren(_) => unimplemented!("insert_origin_var::Pat::Paren"),
        syn::Pat::Path(_) => unimplemented!("insert_origin_var::Pat::Path"),
        syn::Pat::Range(_) => unimplemented!("insert_origin_var::Pat::Range"),
        syn::Pat::Reference(_) => unimplemented!("insert_origin_var::Pat::Reference"),
        syn::Pat::Rest(_) => unimplemented!("insert_origin_var::Pat::Rest"),
        syn::Pat::Slice(_) => unimplemented!("insert_origin_var::Pat::Slice"),
        syn::Pat::Struct(_) => unimplemented!("insert_origin_var::Pat::Struct"),
        syn::Pat::Tuple(_) => unimplemented!("insert_origin_var::Pat::Tuple"),
        syn::Pat::TupleStruct(_) => unimplemented!("insert_origin_var::Pat::TupleStruct"),
        syn::Pat::Type(pat_type) => {
            insert_origin_var(origin_vars, &pat_type.pat);
        }
        syn::Pat::Verbatim(_) => unimplemented!("insert_origin_var::Pat::Verbatim"),
        syn::Pat::Wild(_) => {}
        _ => unimplemented!("insert_origin_var::Pat::Other"),
    }
}

fn new_name_pat(
    pat: &mut syn::Pat,
    stacks: &mut HashMap<String, Vec<usize>>,
    versions: &mut HashMap<String, usize>
) {
    match pat {
        syn::Pat::Const(_) => unimplemented!("rename::new_name_pat::Pat::Const"),
        syn::Pat::Ident(pat_ident) => {
            let new_name = new_name(&pat_ident.ident.to_string(), versions, stacks);
            pat_ident.ident = syn::Ident::new(&new_name, pat_ident.ident.span());
        }
        syn::Pat::Lit(_) => unimplemented!("rename::new_name_pat::Pat::Lit"),
        syn::Pat::Macro(_) => unimplemented!("rename::new_name_pat::Pat::Macro"),
        syn::Pat::Or(_) => unimplemented!("rename::new_name_pat::Pat::Or"),
        syn::Pat::Paren(_) => unimplemented!("rename::new_name_pat::Pat::Paren"),
        syn::Pat::Path(_) => unimplemented!("rename::new_name_pat::Pat::Path"),
        syn::Pat::Range(_) => unimplemented!("rename::new_name_pat::Pat::Range"),
        syn::Pat::Reference(_) => unimplemented!("rename::new_name_pat::Pat::Reference"),
        syn::Pat::Rest(_) => unimplemented!("rename::new_name_pat::Pat::Rest"),
        syn::Pat::Slice(_) => unimplemented!("rename::new_name_pat::Pat::Slice"),
        syn::Pat::Struct(_) => unimplemented!("rename::new_name_pat::Pat::Struct"),
        syn::Pat::Tuple(_) => unimplemented!("rename::new_name_pat::Pat::Tuple"),
        syn::Pat::TupleStruct(_) => unimplemented!("rename::new_name_pat::Pat::TupleStruct"),
        syn::Pat::Type(pat_type) => {
            new_name_pat(&mut pat_type.pat, stacks, versions);
        }
        syn::Pat::Verbatim(_) => unimplemented!("rename::new_name_pat::Pat::Verbatim"),
        syn::Pat::Wild(_) => {}
        _ => unimplemented!("rename::new_name_pat::Pat::Other"),
    }
}

fn rename(
    cfg: &mut CFG,
    node: NodeIndex,
    stacks: &mut HashMap<String, Vec<usize>>,
    versions: &mut HashMap<String, usize>,
    dominators: &Dominators<NodeIndex>
) {
    for phi_function in &mut cfg.graph[node].phi_functions {
        for arg in &mut phi_function.args {
            replace_string(arg, stacks);
        }
        let new_name = new_name(&phi_function.name, versions, stacks);
        phi_function.name = new_name;
    }
    for stmt in &mut cfg.graph[node].statements {
        match &mut stmt.stmt {
            Stmt::Local(local) => {
                if let Some(init) = &mut local.init {
                    replace_vars(&mut init.expr, stacks);
                }
                new_name_pat(&mut local.pat, stacks, versions);
            }
            Stmt::Item(_) => unimplemented!("rename::Stmt::Item"),
            Stmt::Expr(expr, ..) => {
                replace_vars(expr, stacks);
            }
            Stmt::Macro(_) => unimplemented!("rename::Stmt::Macro"),
        }
    }
    let mut new_origin_var_map = HashMap::new();
    let mut all_vars = cfg.graph[node].used_vars.clone();
    all_vars.extend(cfg.graph[node].defined_vars.clone());
    all_vars.extend(cfg.graph[node].assigned_vars.clone());
    for var in all_vars {
        let mut new_key = var.clone();
        replace_string(&mut new_key, stacks);
        new_origin_var_map.insert(new_key, var.clone());
    }
    cfg.graph[node].origin_var_map = new_origin_var_map;
    let succs: Vec<NodeIndex> = cfg.graph
        .neighbors_directed(node, petgraph::Direction::Outgoing)
        .collect();
    // for each succ of current node in the cfg
    //  fill in phi function parameters
    for succ in succs.iter() {
        let mut incomings: Vec<NodeIndex> = cfg.graph
            .neighbors_directed(*succ, petgraph::Direction::Incoming)
            .collect();
        incomings.sort();
        let j = incomings
            .iter()
            .position(|x| *x == node)
            .unwrap_or(0);
        for phi_function in &mut cfg.graph[*succ].phi_functions {
            let origin_var = &phi_function.origin_var;
            if let Some(current_version) = stacks.get(origin_var) {
                if let Some(version) = current_version.last() {
                    let new_var = format!("{}{}", origin_var, version);
                    phi_function.args[j] = new_var.clone();
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
    for var in &cfg.graph[node].origin_vars {
        stacks.get_mut(var).expect(&format!("rename::phi::stacks: {}", var)).pop();
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

fn replace_string(var: &mut String, stacks: &HashMap<String, Vec<usize>>) {
    if let Some(stack) = stacks.get(var) {
        if let Some(current_cnt) = stack.last() {
            *var = format!("{}{}", var, current_cnt);
        }
    }
}

fn new_name(
    var: &str,
    versions: &mut HashMap<String, usize>,
    stacks: &mut HashMap<String, Vec<usize>>
) -> String {
    if let Some(cnt) = versions.get_mut(var) {
        if let Some(stack) = stacks.get_mut(var) {
            stack.push(*cnt);
        }
        let ret = format!("{}{}", var, cnt);
        *cnt += 1;
        ret
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
    fn new_block(&mut self, block_type: BlockType) -> NodeIndex {
        let block = BasicBlock {
            statements: vec![],
            origin_vars: HashSet::new(),
            defined_vars: HashSet::new(),
            used_vars: HashSet::new(),
            block_type,
            live_in: HashSet::new(),
            live_out: HashSet::new(),
            phi_functions: vec![],
            origin_var_map: HashMap::new(),
            assigned_vars: HashSet::new(),
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
        let cond_block = self.new_block(BlockType::IfCond);
        // 创建 then 分支块
        let then_block = self.new_block(BlockType::IfThen);
        // 创建 else 分支块
        let else_block = self.new_block(BlockType::IfElse);
        // 创建合并块
        let merge_block = self.new_block(BlockType::Normal);

        // 连接当前块到条件检查块
        self.connect_to(cond_block);
        self.set_current_block(cond_block);
        let mut visitor = UseDefineVisitor::new();
        visitor.visit_expr(&expr_if.cond);
        self.visit_expr(&expr_if.cond);
        self.cfg.graph[cond_block].defined_vars.extend(visitor.define_vars.drain());
        self.cfg.graph[cond_block].used_vars.extend(visitor.used_vars.drain());
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
        let loop_block = self.new_block(BlockType::LoopBody);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);

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
        let new_block = self.new_block(BlockType::Normal);
        self.set_current_block(new_block);
    }

    // 处理 continue 语句
    fn handle_continue(&mut self, _expr_continue: &syn::ExprContinue) {
        if let Some(_) = self.loop_stack.last() {
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
        let new_block = self.new_block(BlockType::Normal);
        self.set_current_block(new_block);
    }

    // 处理 match 语句
    fn handle_match(&mut self, expr_match: &syn::ExprMatch) {
        // 创建合并块
        let merge_block = self.new_block(BlockType::Normal);

        for arm in &expr_match.arms {
            // 为每个匹配分支创建一个块
            let arm_block = self.new_block(BlockType::Normal);
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
        let condition_block = self.new_block(BlockType::WhileCond);
        // 创建循环体块
        let loop_block = self.new_block(BlockType::WhileBody);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);

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
        let init_block = self.new_block(BlockType::ForInit);
        // 创建条件检查块
        let condition_block = self.new_block(BlockType::ForCond);
        // 创建循环体块
        let loop_block = self.new_block(BlockType::ForBody);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);

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
        let new_block = self.new_block(BlockType::Normal);
        for arg in &i.sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => {}
                syn::FnArg::Typed(pat_type) => {
                    let ty = pat_type.ty.as_ref();
                    match &*pat_type.pat {
                        syn::Pat::Ident(pat_ident) => {
                            if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                                let local = syn
                                    ::parse2(
                                        quote::quote! {
                                    let #pat_ident: #ty;
                                }
                                    )
                                    .unwrap();
                                block.statements.push(CustomStmt { stmt: local });
                                block.defined_vars.insert(pat_ident.ident.to_string());
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

    fn visit_expr_while(&mut self, i: &'ast syn::ExprWhile) {
        self.handle_while_loop(i);
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
        if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
            if let Some(last) = block.statements.last() {
                let mut collector = UseDefineVisitor::new();
                collector.visit_stmt(&last.stmt);
                block.used_vars.extend(collector.used_vars);
                block.defined_vars.extend(collector.define_vars);
                block.assigned_vars.extend(collector.assigned_vars);
            }
        }
    }
}

// fn replace_phi_nodes(cfg: &mut CFG) {
//     let mut stacks = HashMap::new();
// }

// fn insert_copies(cfg: &mut CFG) {

// }
