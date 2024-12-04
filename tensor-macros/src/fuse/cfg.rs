use std::collections::{ HashMap, HashSet };
use std::rc::Rc;

use petgraph::algo::dominators::Dominators;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use quote::ToTokens;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::Stmt;

use crate::fuse::controlflow_detector::ControlFlowDetector;

use super::{ codegen, expr_ty };
use super::expr_call_use_visitor::ExprCallUseVisitor;
use super::expr_expand::ExprExpander;
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
    ExprBlockAssign,
    ExprBlock,
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

#[derive(Clone)]
pub(crate) struct BasicBlock {
    pub(crate) statements: Vec<CustomStmt>,
    pub(crate) phi_functions: Vec<PhiFunction>,
    pub(crate) defined_vars: HashSet<syn::Ident>,
    pub(crate) assigned_vars: HashSet<syn::Ident>,
    pub(crate) used_vars: HashSet<syn::Ident>,
    pub(crate) block_type: BlockType,
    pub(crate) live_out: HashSet<syn::Ident>,
    pub(crate) origin_var_map: HashMap<syn::Ident, syn::Ident>,
}

impl std::fmt::Debug for BasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasicBlock")
            .field("statements", &self.statements)
            .field("phi_functions", &self.phi_functions)
            .field(
                "defined_vars",
                &self.defined_vars
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            )
            .field(
                "assigned_vars",
                &self.assigned_vars
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            )
            .field(
                "used_vars",
                &self.used_vars
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            )
            .field("block_type", &self.block_type)
            .field(
                "live_out",
                &self.live_out
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            )
            .field(
                "origin_var_map",
                &self.origin_var_map
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect::<HashMap<_, _>>()
            )
            .finish()
    }
}

// CFG 结构
pub(crate) struct CFG {
    pub(crate) graph: Graph<BasicBlock, ()>,
    pub(crate) block_id: BlockId,
    pub(crate) entry: NodeIndex,
}

fn visit_expr_assign_add_var(
    expr: &syn::ExprAssign,
    definitions: &mut HashMap<syn::Ident, Vec<NodeIndex>>,
    node: NodeIndex
) {
    match expr.left.as_ref() {
        syn::Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                definitions.entry(ident.clone()).or_insert_with(Vec::new).push(node);
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

fn check_expr_assign(expr: &syn::ExprAssign, var: &syn::Ident, status: &mut VarStatus) {
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
                if ident == var {
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
            defined_vars: HashSet::new(),
            used_vars: HashSet::new(),
            live_out: HashSet::new(),
            block_type: BlockType::Normal,
            phi_functions: vec![],
            origin_var_map: HashMap::new(),
            assigned_vars: HashSet::new(),
        };
        let entry = graph.add_node(entry_block);
        CFG { graph, entry, block_id: BlockId::new(entry) }
    }

    pub(crate) fn live_analysis(&mut self, type_table: &HashMap<syn::Ident, Type>) {
        let mut live_in = HashMap::new();
        let mut live_out = HashMap::new();
        let mut used_vars = HashMap::new();
        let mut defined_vars = HashMap::new();

        for block in self.graph.node_indices() {
            let block_data = &self.graph[block];
            let mut use_define_visitor = UseDefineVisitor::new();
            for stmt in &block_data.statements {
                use_define_visitor.visit_stmt(&stmt.stmt);
            }
            used_vars.insert(block, use_define_visitor.used_vars);
            defined_vars.insert(block, use_define_visitor.define_vars);
            for phi in &block_data.phi_functions {
                for arg in &phi.args {
                    used_vars.entry(block).or_insert_with(HashSet::new).insert(arg.clone());
                }
                defined_vars.entry(block).or_insert_with(HashSet::new).insert(phi.name.clone());
            }
        }

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
                let mut new_out = HashSet::new();
                for succ in self.graph.neighbors_directed(block, petgraph::Direction::Outgoing) {
                    new_out.extend(live_in[&succ].iter().cloned());
                }

                // 计算IN[B] = USE[B] ∪ (OUT[B] - DEF[B])
                let mut new_in = used_vars[&block].clone();
                for var in new_out.iter() {
                    if !defined_vars[&block].contains(var) {
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
            self.graph[block].live_out = live_out[&block].clone();

            // check if current block has return statement
            if self.graph[block].block_type == BlockType::Normal {
                let mut extra_live_outs = HashSet::new();
                for stmt in &self.graph[block].statements {
                    let mut visitor = ExprCallUseVisitor::new(type_table);
                    visitor.visit_stmt(&stmt.stmt);
                    extra_live_outs.extend(visitor.used_vars.drain());
                }
                self.graph[block].live_out.extend(extra_live_outs.drain());
            }
        }
    }

    pub(crate) fn add_block(&mut self, block: BasicBlock) -> NodeIndex {
        self.graph.add_node(block)
    }

    pub(crate) fn connect(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, ());
    }

    pub(crate) fn get_variable_definitions(&self) -> HashMap<syn::Ident, Vec<NodeIndex>> {
        let mut definitions: HashMap<syn::Ident, Vec<NodeIndex>> = HashMap::new();

        for node in self.graph.node_indices() {
            for stmt in &self.graph[node].statements {
                match &stmt.stmt {
                    Stmt::Local(local) => {
                        let mut use_define_visitor = UseDefineVisitor::new();
                        use_define_visitor.visit_pat(&local.pat);
                        for var in use_define_visitor.define_vars.drain() {
                            definitions.entry(var).or_insert_with(Vec::new).push(node);
                        }
                        for var in use_define_visitor.used_vars.drain() {
                            definitions.entry(var).or_insert_with(Vec::new).push(node);
                        }
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
        variable_definitions: &HashMap<syn::Ident, Vec<NodeIndex>>
    ) {
        let mut has_already = HashMap::new();

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
                                                if &pat_ident.ident == var {
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
                                self.graph[*frontier].used_vars.insert(var.clone());
                                self.graph[*frontier].phi_functions.push(phi_function);
                                self.graph[*frontier].defined_vars.insert(var.clone());
                            }
                            has_already.get_mut(var).unwrap().insert(*frontier);

                            // **只有当变量 v 不在 frontier 的 orig 中时，才将 frontier 加入工作集**
                            if !self.graph[*frontier].defined_vars.contains(var) {
                                work.push(*frontier);
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn build_graphs(
        &self,
        type_table: HashMap<syn::Ident, Type>
    ) -> Graph<super::build_graph::Graph, ()> {
        let table = Rc::new(type_table);
        let mut graph = Graph::<super::build_graph::Graph, ()>::new();
        let mut sorted_indices = self.graph.node_indices().collect::<Vec<_>>();
        sorted_indices.sort();
        for node in sorted_indices {
            let mut comp_graph = super::build_graph::Graph::new(table.clone(), node.index());
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

    pub(crate) fn add_extra_temps(&mut self, graph: &Graph<super::build_graph::Graph, ()>) {
        for node in graph.node_indices() {
            let graph = graph.node_weight(node).expect("graph weight not found");
            for temp in graph.extra_temps.iter() {
                self.graph[node].origin_var_map.insert(temp.clone(), temp.clone());
            }
        }
    }

    // 变量重命名
    pub(crate) fn rename_variables(&mut self, dominators: &Dominators<NodeIndex>) {
        let mut stacks = HashMap::new();
        let mut versions = HashMap::new();

        // 收集所有变量
        let variables: HashSet<syn::Ident> = self.graph
            .node_indices()
            .flat_map(|node| {
                let mut vars = HashSet::new();
                vars.extend(self.graph[node].defined_vars.clone());
                vars.extend(self.graph[node].used_vars.clone());
                vars.extend(self.graph[node].assigned_vars.clone());
                vars
            })
            .collect();

        // 初始化栈和版本
        for var in variables {
            stacks.insert(var.clone(), Vec::new());
            versions.insert(var, 0);
        }

        rename(self, self.entry, &mut stacks, &mut versions, dominators);
    }

    pub(crate) fn gen_code(&mut self) -> crate::TokenStream2 {
        let block_id = core::mem::take(&mut self.block_id);
        let mut child_code = quote::quote!();
        for child in block_id.children.into_iter() {
            child_code.extend(self._gen_code(child));
        }
        child_code
    }

    fn _gen_code(&self, block_id: BlockId) -> crate::TokenStream2 {
        let mut body = quote::quote!();
        let block = &self.graph[block_id.id];
        let code = codegen::stmt(block);
        let mut child_code = quote::quote!();
        for child in block_id.children.into_iter() {
            child_code.extend(self._gen_code(child));
        }
        match block.block_type {
            BlockType::Normal => {
                body.extend(quote::quote!(#code #child_code));
            }
            BlockType::IfCond => {
                body.extend(quote::quote!(if #code #child_code));
            }
            BlockType::IfThen => {
                body.extend(quote::quote!({ #code #child_code }));
            }
            BlockType::IfElse => {
                body.extend(quote::quote!(else { #code #child_code }));
            }
            BlockType::ForInit => {
                body.extend(quote::quote!(for #code #child_code));
            }
            BlockType::ForBody => {
                body.extend(quote::quote!({#code #child_code}));
            }
            BlockType::ForCond => {
                body.extend(quote::quote!(in #code #child_code));
            }
            BlockType::WhileCond => {
                body.extend(quote::quote!(while #code #child_code));
            }
            BlockType::WhileBody => {
                body.extend(quote::quote!({#code #child_code}));
            }
            BlockType::LoopBody => {
                body.extend(quote::quote!(loop {#code #child_code}));
            }
            BlockType::ExprBlock => {
                body.extend(quote::quote!({#code #child_code};));
            }
            BlockType::ExprBlockAssign => {
                body.extend(quote::quote!(let #code #child_code =));
            }
        }
        body
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

fn new_name_pat(
    pat: &mut syn::Pat,
    stacks: &mut HashMap<syn::Ident, Vec<usize>>,
    versions: &mut HashMap<syn::Ident, usize>,
    new_origin_var_map: &mut HashMap<syn::Ident, syn::Ident>
) {
    match pat {
        syn::Pat::Const(_) => unimplemented!("rename::new_name_pat::Pat::Const"),
        syn::Pat::Ident(pat_ident) => {
            pat_ident.ident = new_name(&pat_ident.ident, versions, stacks, new_origin_var_map);
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
        syn::Pat::Tuple(tuple) => {
            for pat in tuple.elems.iter_mut() {
                new_name_pat(pat, stacks, versions, new_origin_var_map);
            }
        }
        syn::Pat::TupleStruct(_) => unimplemented!("rename::new_name_pat::Pat::TupleStruct"),
        syn::Pat::Type(pat_type) => {
            new_name_pat(&mut pat_type.pat, stacks, versions, new_origin_var_map);
        }
        syn::Pat::Verbatim(_) => unimplemented!("rename::new_name_pat::Pat::Verbatim"),
        syn::Pat::Wild(_) => {}
        _ => unimplemented!("rename::new_name_pat::Pat::Other"),
    }
}

fn rename(
    cfg: &mut CFG,
    node: NodeIndex,
    stacks: &mut HashMap<syn::Ident, Vec<usize>>,
    versions: &mut HashMap<syn::Ident, usize>,
    dominators: &Dominators<NodeIndex>
) {
    let mut new_origin_var_map = cfg.graph[node].origin_var_map.clone();
    for phi_function in &mut cfg.graph[node].phi_functions {
        for arg in &mut phi_function.args {
            replace_string(arg, stacks);
        }
        let new_name = new_name(&phi_function.name, versions, stacks, &mut new_origin_var_map);
        phi_function.name = new_name;
    }
    for stmt in &mut cfg.graph[node].statements {
        match &mut stmt.stmt {
            Stmt::Local(local) => {
                if let Some(init) = &mut local.init {
                    replace_vars(&mut init.expr, stacks, &mut new_origin_var_map);
                }
                new_name_pat(&mut local.pat, stacks, versions, &mut new_origin_var_map);
            }
            Stmt::Item(_) => unimplemented!("rename::Stmt::Item"),
            Stmt::Expr(expr, ..) => {
                replace_vars(expr, stacks, &mut new_origin_var_map);
            }
            Stmt::Macro(mc) => {
                let tokens = mc.mac.tokens.clone();
                let mut new_tokens = proc_macro2::TokenStream::new();
                for token in tokens.into_iter() {
                    if let proc_macro2::TokenTree::Ident(ident) = token {
                        if let Some(stack) = stacks.get(&ident) {
                            if let Some(version) = stack.last() {
                                let new_ident = syn::Ident::new(
                                    &format!("{}{}", ident, version),
                                    ident.span()
                                );
                                new_origin_var_map.insert(new_ident.clone(), ident);
                                new_tokens.extend(quote::quote!(#new_ident));
                            }
                        }
                    } else {
                        new_tokens.extend(quote::quote!(#token));
                    }
                }
                mc.mac.tokens = new_tokens;
            }
        }
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
                    phi_function.args[j] = syn::Ident::new(
                        &format!("{}{}", origin_var, version),
                        origin_var.span()
                    );
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
    for var in &cfg.graph[node].defined_vars {
        stacks.get_mut(var).expect(&format!("rename::phi::stacks: {}", var)).pop();
    }
}

fn replace_vars(
    expr: &mut syn::Expr,
    stacks: &HashMap<syn::Ident, Vec<usize>>,
    new_origin_var_map: &mut HashMap<syn::Ident, syn::Ident>
) {
    struct VarRenamer<'a> {
        stacks: &'a HashMap<syn::Ident, Vec<usize>>,
        new_origin_var_map: &'a mut HashMap<syn::Ident, syn::Ident>,
    }
    impl<'a> syn::visit_mut::VisitMut for VarRenamer<'a> {
        fn visit_expr_mut(&mut self, node: &mut syn::Expr) {
            match node {
                syn::Expr::Path(expr_path) => {
                    if expr_path.qself.is_none() && expr_path.path.segments.len() == 1 {
                        let var = expr_path.path.segments[0].ident.clone();
                        if let Some(stack) = self.stacks.get(&var) {
                            if let Some(current_cnt) = stack.last() {
                                expr_path.path.segments[0].ident = syn::Ident::new(
                                    &format!("{}{}", var, current_cnt),
                                    expr_path.path.segments[0].ident.span()
                                );
                                self.new_origin_var_map.insert(
                                    syn::Ident::new(
                                        &format!("{}{}", var, current_cnt),
                                        expr_path.path.segments[0].ident.span()
                                    ),
                                    var
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
    syn::visit_mut::visit_expr_mut(&mut (VarRenamer { stacks, new_origin_var_map }), expr);
}

fn replace_string(var: &mut syn::Ident, stacks: &HashMap<syn::Ident, Vec<usize>>) {
    if let Some(stack) = stacks.get(var) {
        if let Some(current_cnt) = stack.last() {
            *var = syn::Ident::new(&format!("{}{}", var, current_cnt), var.span());
        }
    }
}

fn new_name(
    var: &syn::Ident,
    versions: &mut HashMap<syn::Ident, usize>,
    stacks: &mut HashMap<syn::Ident, Vec<usize>>,
    new_origin_var_map: &mut HashMap<syn::Ident, syn::Ident>
) -> syn::Ident {
    if let Some(cnt) = versions.get_mut(var) {
        if let Some(stack) = stacks.get_mut(var) {
            stack.push(*cnt);
        }
        let ret = syn::Ident::new(&format!("{}{}", var, cnt), var.span());
        *cnt += 1;
        new_origin_var_map.insert(ret.clone(), var.clone());
        ret
    } else {
        panic!("{} not found in new_name", var);
    }
}

#[derive(Clone, Default)]
pub(crate) struct BlockId {
    pub(crate) children: Vec<BlockId>,
    pub(crate) id: NodeIndex,
}

impl std::fmt::Debug for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.children.is_empty() {
            f.debug_struct("BlockId")
                .field("children", &self.children)
                .field("id", &self.id)
                .finish()
        } else {
            write!(f, "{:?}", self.id)
        }
    }
}

impl BlockId {
    pub(crate) fn new(id: NodeIndex) -> Self {
        BlockId { children: vec![], id }
    }
}

pub(crate) struct CFGBuilder<'a> {
    pub(crate) cfg: &'a mut CFG,
    pub(crate) current_block: NodeIndex,
    pub(crate) block_ids: BlockId,
    pub(crate) current_expr: Option<syn::Expr>,
    pub(crate) global_block_cnt: usize,
}

impl<'a> CFGBuilder<'a> {
    pub(crate) fn new(cfg: &'a mut CFG) -> Self {
        let entry = cfg.entry.clone();
        CFGBuilder {
            cfg,
            current_block: entry,
            block_ids: BlockId { children: vec![], id: entry },
            current_expr: None,
            global_block_cnt: 0,
        }
    }

    // 创建新的基本块并返回其索引
    fn new_block(&mut self, block_type: BlockType) -> NodeIndex {
        let block = BasicBlock {
            statements: vec![],
            defined_vars: HashSet::new(),
            used_vars: HashSet::new(),
            block_type,
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

    fn set_current_block_id(&mut self, new_block_id: BlockId) {
        self.block_ids = new_block_id;
    }

    // 处理 if 语句
    fn handle_if(&mut self, expr_if: &syn::ExprIf) {
        let cond_block = self.new_block(BlockType::IfCond);
        let cond_block_id = BlockId::new(cond_block);
        // 创建 then 分支块
        let then_block = self.new_block(BlockType::IfThen);
        let then_block_id = BlockId::new(then_block);
        // 创建 else 分支块
        let else_block = self.new_block(BlockType::IfElse);
        let else_block_id = BlockId::new(else_block);
        // 创建合并块
        let merge_block = self.new_block(BlockType::Normal);
        let merge_block_id = BlockId::new(merge_block);

        let mut current_block_id = core::mem::take(&mut self.block_ids);
        // 连接当前块到条件检查块
        self.connect_to(cond_block);
        self.set_current_block(cond_block);
        self.set_current_block_id(cond_block_id);

        let mut visitor = UseDefineVisitor::new();
        visitor.visit_expr(&expr_if.cond);
        self.visit_expr(&expr_if.cond);
        let cond_block_id = core::mem::take(&mut self.block_ids);
        self.cfg.graph[cond_block].defined_vars.extend(visitor.define_vars.drain());
        self.cfg.graph[cond_block].used_vars.extend(visitor.used_vars.drain());
        // 连接当前块到 then 和 else 分支
        self.connect_to(then_block);
        self.connect_to(else_block);

        self.set_current_block_id(then_block_id);
        // 处理 then 分支
        self.set_current_block(then_block);
        self.visit_block(&expr_if.then_branch);
        let then_block_id = core::mem::take(&mut self.block_ids);
        self.connect_to(merge_block);

        // 处理 else 分支
        self.set_current_block_id(else_block_id);
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
        let else_block_id = core::mem::take(&mut self.block_ids);
        self.connect_to(merge_block);

        current_block_id.children.push(cond_block_id);
        current_block_id.children.push(then_block_id);
        current_block_id.children.push(else_block_id);
        current_block_id.children.push(merge_block_id);
        // 更新当前块为合并块
        self.set_current_block(merge_block);
        self.set_current_block_id(current_block_id);
    }

    // 处理 loop 语句
    fn handle_loop(&mut self, expr_loop: &syn::ExprLoop) {
        // 创建循环体块
        let loop_block = self.new_block(BlockType::LoopBody);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);

        // 连接当前块到循环体块
        self.connect_to(loop_block);

        // 设置当前块为循环体块，并处理循环体
        self.set_current_block(loop_block);
        self.visit_block(&expr_loop.body);

        // 连接循环体块回到自身，表示下一次迭代
        self.connect_to(loop_block);
        // 连接循环体块到循环后的块，表示退出循环
        self.connect_to(after_loop_block);

        // 更新当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
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

    fn handle_block(&mut self, block: &syn::ExprBlock) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let assign_block = self.new_block(BlockType::ExprBlockAssign);
        let assign_block_id = BlockId::new(assign_block);
        let new_block = self.new_block(BlockType::ExprBlock);
        let new_block_id = BlockId::new(new_block);

        let merge_block = self.new_block(BlockType::Normal);
        let merge_block_id = BlockId::new(merge_block);

        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let block_ident = syn::Ident::new(
            &format!("__block_out_{}", self.global_block_cnt),
            block.span()
        );
        self.cfg.graph[assign_block].statements.push(CustomStmt {
            stmt: syn
                ::parse2(quote::quote! {
                let #block_ident;
            })
                .expect("cfg_builder::handle_block::assign_block"),
        });

        self.connect_to(new_block);
        self.set_current_block(new_block);
        self.set_current_block_id(new_block_id);
        self.visit_block(&block.block);
        let new_block_id = core::mem::take(&mut self.block_ids);
        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(new_block_id);
        current_block_id.children.push(merge_block_id);
        self.connect_to(merge_block);
        self.set_current_block_id(current_block_id);
        self.set_current_block(merge_block);
    }

    fn handle_while_loop(&mut self, expr_while: &syn::ExprWhile) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        // 创建条件检查块
        let condition_block = self.new_block(BlockType::WhileCond);
        let condition_block_id = BlockId::new(condition_block);
        // 创建循环体块
        let loop_block = self.new_block(BlockType::WhileBody);
        let loop_block_id = BlockId::new(loop_block);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);
        let after_loop_block_id = BlockId::new(after_loop_block);

        // 连接当前块到条件检查块
        self.connect_to(condition_block);

        // 连接条件检查块到循环体块（如果条件为真）和循环后的块（如果条件为假）
        self.set_current_block(condition_block);
        self.set_current_block_id(condition_block_id);
        if let Some(block) = self.cfg.graph.node_weight_mut(condition_block) {
            // 你可以将条件表达式添加到条件检查块中
            block.statements.push(CustomStmt { stmt: Stmt::Expr(*expr_while.cond.clone(), None) });
        }
        let condition_block_id = core::mem::take(&mut self.block_ids);

        // 创建两个新的连接：条件为真和条件为假
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_while.body);
        let loop_block_id = core::mem::take(&mut self.block_ids);

        // 连接循环体块回到条件检查块（表示下一次迭代）
        self.connect_to(condition_block);
        // 连接循环体块到循环后的块（如果有 break）
        self.connect_to(after_loop_block);

        // 设置当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
        current_block_id.children.push(condition_block_id);
        current_block_id.children.push(loop_block_id);
        current_block_id.children.push(after_loop_block_id);
        self.set_current_block_id(current_block_id);
    }

    fn handle_for_loop(&mut self, expr_for: &syn::ExprForLoop) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        // 创建迭代器初始化块
        let init_block = self.new_block(BlockType::ForInit);
        let init_block_id = BlockId::new(init_block);
        // 创建条件检查块
        let condition_block = self.new_block(BlockType::ForCond);
        let condition_block_id = BlockId::new(condition_block);
        // 创建循环体块
        let loop_block = self.new_block(BlockType::ForBody);
        let loop_block_id = BlockId::new(loop_block);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);
        let after_loop_block_id = BlockId::new(after_loop_block);

        // 连接当前块到初始化块
        self.connect_to(init_block);

        // 处理迭代器初始化和元素绑定
        self.set_current_block(init_block);
        self.set_current_block_id(init_block_id);
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
        let init_block_id = core::mem::take(&mut self.block_ids);
        // 连接初始化块到条件检查块
        self.connect_to(condition_block);

        // 处理条件检查
        self.set_current_block(condition_block);
        self.set_current_block_id(condition_block_id);
        if let Some(block) = self.cfg.graph.node_weight_mut(condition_block) {
            let expr = &expr_for.expr;
            block.statements.push(CustomStmt { stmt: Stmt::Expr(*(*expr).clone(), None) });
        }
        let condition_block_id = core::mem::take(&mut self.block_ids);
        // 创建连接：条件为真进入循环体块，条件为假进入循环后的块
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_for.body);
        let loop_block_id = core::mem::take(&mut self.block_ids);
        // 连接循环体块回到条件检查块（表示下一次迭代）
        self.connect_to(condition_block);
        // 连接循环体块到循环后的块（如果有 break）
        self.connect_to(after_loop_block);

        // 设置当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
        current_block_id.children.push(init_block_id);
        current_block_id.children.push(condition_block_id);
        current_block_id.children.push(loop_block_id);
        current_block_id.children.push(after_loop_block_id);
        self.set_current_block_id(current_block_id);
    }
}

impl<'ast, 'a> Visit<'ast> for CFGBuilder<'a> {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let new_block = self.new_block(BlockType::Normal);
        let new_block_id = BlockId::new(new_block);
        for arg in &i.sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => {}
                syn::FnArg::Typed(pat_type) => {
                    let ty = pat_type.ty.as_ref();
                    let mut vars_collector = UseDefineVisitor::new();
                    vars_collector.visit_pat(&pat_type.pat);
                    if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                        for var in vars_collector.define_vars {
                            block.defined_vars.insert(var.clone());
                        }
                        for var in vars_collector.used_vars {
                            block.defined_vars.insert(var.clone());
                        }
                        let pat = &pat_type.pat;
                        let local =
                            quote::quote! {
                            let #pat: #ty;
                        };
                        block.statements.push(CustomStmt {
                            stmt: syn::parse2(local).expect("cfg::visit_item_fn::local"),
                        });
                    }
                }
            }
        }
        self.connect_to(new_block);
        self.set_current_block(new_block);
        self.set_current_block_id(new_block_id);
        self.visit_block(&i.block);
        let new_block_id = core::mem::take(&mut self.block_ids);
        current_block_id.children.push(new_block_id);
        self.set_current_block_id(current_block_id);
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

    fn visit_expr_block(&mut self, block: &'ast syn::ExprBlock) {
        self.handle_block(block);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let mut new_call = call.clone();
        for arg in &mut new_call.args {
            match arg {
                syn::Expr::Block(expr_block) => {
                    self.visit_expr_block(expr_block);
                    let ident = syn::Ident::new(
                        &format!("__block_out_{}", self.global_block_cnt),
                        expr_block.span()
                    );
                    *arg = syn
                        ::parse2(quote::quote! {
                        #ident
                    })
                        .expect("cfg_builder::visit_expr_call::block_out");
                    self.global_block_cnt += 1;
                }
                _ =>
                    unimplemented!(
                        "cfg_builder::visit_expr_call::{:?}",
                        expr_ty::ExprType::from(arg)
                    ),
            }
        }
        self.current_expr = Some(syn::Expr::Call(new_call));
    }
    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let mut new_call = i.clone();
        let mut detector = ControlFlowDetector::new();
        detector.visit_expr(&new_call.receiver);
        if detector.has_control_flow {
            match &mut new_call.receiver.as_mut() {
                syn::Expr::Block(expr_block) => {
                    self.visit_expr_block(expr_block);
                    let ident = syn::Ident::new(
                        &format!("__block_out_{}", self.global_block_cnt),
                        expr_block.span()
                    );
                    new_call.receiver = Box::new(
                        syn
                            ::parse2(
                                quote::quote! {
                        #ident
                    }
                            )
                            .expect("cfg_builder::visit_expr_method_call::block_out")
                    );
                    self.global_block_cnt += 1;
                }
                syn::Expr::Paren(expr_paren) => {
                    
                }
                _ => {
                    panic!("cfg_builder::visit_expr_method_call::receiver");
                }
            }
        } else {
            let mut expander = ExprExpander::new();
            expander.visit_expr(&new_call.receiver);
            for stmt in expander.stmts {
                self.visit_stmt(&stmt);
            }
            if let Some(expr) = expander.current_expr {
                new_call.receiver = Box::new(expr);
            }
        }
        for arg in &mut new_call.args {
            match arg {
                syn::Expr::Block(expr_block) => {
                    self.visit_expr_block(expr_block);
                    let ident = syn::Ident::new(
                        &format!("__block_out_{}", self.global_block_cnt),
                        expr_block.span()
                    );
                    *arg = syn
                        ::parse2(quote::quote! {
                        #ident
                    })
                        .expect("cfg_builder::visit_expr_call::block_out");
                    self.global_block_cnt += 1;
                }
                _ =>
                    unimplemented!(
                        "cfg_builder::visit_expr_call::{:?}",
                        expr_ty::ExprType::from(arg)
                    ),
            }
        }
        self.current_expr = Some(syn::Expr::MethodCall(new_call));
    }

    fn visit_local(&mut self, i: &'ast syn::Local) {
        if let Some(init) = &i.init {
            self.visit_expr(&init.expr);
            let left = &i.pat;
            let expr = self.current_expr
                .take()
                .expect(&format!("cfg_builder::visit_local::expr::{:#?}", self.current_expr));
            if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                block.statements.push(CustomStmt {
                    stmt: syn
                        ::parse2(
                            quote::quote! {
                        let #left = #expr;
                    }
                        )
                        .expect("cfg_builder::visit_local::init"),
                });
            }
        } else {
            panic!("cfg_builder::visit_local::init");
        }
    }

    // fn visit_macro(&mut self, i: &'ast syn::Macro) {

    // }

    // fn visit_item(&mut self, i: &'ast syn::Item) {

    // }

    fn visit_expr(&mut self, node: &'ast syn::Expr) {
        let mut detector = ControlFlowDetector::new();
        detector.visit_expr(node);
        if detector.has_control_flow {
            syn::visit::visit_expr(self, node);
        } else {
            if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                let mut expander = ExprExpander::new();
                expander.visit_stmt(&syn::Stmt::Expr(node.clone(), None));
                for stmt in expander.stmts {
                    let mut collector = UseDefineVisitor::new();
                    collector.visit_stmt(&stmt);
                    block.used_vars.extend(collector.used_vars);
                    block.defined_vars.extend(collector.define_vars);
                    block.assigned_vars.extend(collector.assigned_vars);
                    block.statements.push(CustomStmt { stmt });
                }
            }
        }
    }

    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        println!("stmt: {:#?}", node.to_token_stream().to_string());
        let mut detector = ControlFlowDetector::new();
        detector.visit_stmt(node);
        if detector.has_control_flow {
            println!("has control flow");
            syn::visit::visit_stmt(self, node);
        } else {
            if let Some(block) = self.cfg.graph.node_weight_mut(self.current_block) {
                let mut expander = ExprExpander::new();
                expander.visit_stmt(node);
                for stmt in expander.stmts {
                    let mut collector = UseDefineVisitor::new();
                    collector.visit_stmt(&stmt);
                    block.used_vars.extend(collector.used_vars);
                    block.defined_vars.extend(collector.define_vars);
                    block.assigned_vars.extend(collector.assigned_vars);
                    block.statements.push(CustomStmt { stmt });
                }
            }
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
