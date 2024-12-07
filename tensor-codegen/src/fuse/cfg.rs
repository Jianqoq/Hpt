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

use super::{ codegen, expr_ty };
use super::errors::Error;
use super::expr_call_use_visitor::ExprCallUseVisitor;
use super::phi_function::PhiFunction;
use super::ty_infer::Type;
use super::use_define_visitor::UseDefineVisitor;
use super::var_recover::VarRecover;

#[derive(Clone)]
pub(crate) struct CustomStmt {
    pub(crate) stmt: syn::Stmt,
}

#[derive(Clone, PartialEq)]
pub(crate) enum BlockType {
    Normal,
    IfAssign,
    IfCond,
    IfThen,
    IfThenEnd,
    IfElseEnd,
    ElseIfCond,
    ForAssign,
    ForInit,
    ForBody,
    ForCond,
    WhileAssign,
    WhileCond,
    WhileBody,
    LoopAssign,
    LoopBody,
    ExprBlockAssign,
    ExprBlock,
    ClosureArgs,
    ClosureBody,
    ClosureAssign,
    FnArgs,
    FnVisibility(syn::Visibility),
    FnName,
    FnRet(syn::ReturnType),
    Generics(syn::Generics),
    FnBody,
    Where(syn::WhereClause),
}

impl std::fmt::Debug for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::IfAssign => write!(f, "IfAssign"),
            Self::IfCond => write!(f, "IfCond"),
            Self::IfThen => write!(f, "IfThen"),
            Self::IfThenEnd => write!(f, "IfThenEnd"),
            Self::IfElseEnd => write!(f, "IfElseEnd"),
            Self::ElseIfCond => write!(f, "ElseIfCond"),
            Self::ForAssign => write!(f, "ForAssign"),
            Self::ForInit => write!(f, "ForInit"),
            Self::ForBody => write!(f, "ForBody"),
            Self::ForCond => write!(f, "ForCond"),
            Self::WhileAssign => write!(f, "WhileAssign"),
            Self::WhileCond => write!(f, "WhileCond"),
            Self::WhileBody => write!(f, "WhileBody"),
            Self::LoopAssign => write!(f, "LoopAssign"),
            Self::LoopBody => write!(f, "LoopBody"),
            Self::ExprBlockAssign => write!(f, "ExprBlockAssign"),
            Self::ExprBlock => write!(f, "ExprBlock"),
            Self::ClosureArgs => write!(f, "ClosureArgs"),
            Self::ClosureBody => write!(f, "ClosureBody"),
            Self::ClosureAssign => write!(f, "ClosureAssign"),
            Self::FnArgs => write!(f, "FnArgs"),
            Self::FnVisibility(_) => write!(f, "FnVisibility"),
            Self::FnName => write!(f, "FnName"),
            Self::FnRet(_) => write!(f, "FnRet"),
            Self::Generics(_) => write!(f, "Generics"),
            Self::FnBody => write!(f, "FnBody"),
            Self::Where(_) => write!(f, "Where"),
        }
    }
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
    pub(crate) errors: Vec<Error>,
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

fn check_expr_assign(
    errors: &mut Vec<Error>,
    expr: &syn::ExprAssign,
    var: &syn::Ident,
    status: &mut VarStatus
) {
    match &expr.left.as_ref() {
        syn::Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                if ident == var {
                    *status = VarStatus::Assigned;
                }
            }
        }
        _ => {
            errors.push(
                Error::Unsupported(
                    expr.span(),
                    "check_expr_assign",
                    format!("{:?}", expr_ty::ExprType::from(expr.left.as_ref()))
                )
            );
        }
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
        CFG { graph, entry, block_id: BlockId::new(entry), errors: vec![] }
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

    /// analyze the variables being used in the inter-block
    pub(crate) fn inter_live_analysis(&mut self) {
        for block in self.graph.node_indices() {
            let block_data = &mut self.graph[block];
            let mut record = HashSet::new();
            let mut live_out = HashSet::new();
            for stmt in &block_data.statements {
                let mut use_define_visitor = UseDefineVisitor::new();
                use_define_visitor.visit_stmt(&stmt.stmt);
                for used in use_define_visitor.used_vars
                    .drain()
                    .chain(use_define_visitor.assigned_vars.drain()) {
                    if record.contains(&used) {
                        live_out.insert(used);
                    } else {
                        record.insert(used);
                    }
                }
            }
            block_data.live_out.extend(live_out.drain());
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
        let mut errors = Vec::new();
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
                                                check_expr_assign(
                                                    &mut errors,
                                                    assign,
                                                    var,
                                                    &mut status
                                                );
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
        self.errors.extend(errors);
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
    pub(crate) fn rename_variables(
        &mut self,
        dominators: &Dominators<NodeIndex>
    ) -> anyhow::Result<()> {
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

        rename(self, self.entry, &mut stacks, &mut versions, dominators)?;
        Ok(())
    }

    pub(crate) fn fill_variables(&mut self) {
        for node in self.graph.node_indices() {
            let block = &mut self.graph[node];
            for stmt in &mut block.statements {
                let mut use_define_visitor = UseDefineVisitor::new();
                use_define_visitor.visit_stmt(&stmt.stmt);
                for var in use_define_visitor.used_vars.drain() {
                    block.used_vars.insert(var);
                }
                for var in use_define_visitor.define_vars.drain() {
                    block.defined_vars.insert(var);
                }
                for assigned_var in use_define_visitor.assigned_vars.drain() {
                    block.assigned_vars.insert(assigned_var);
                }
            }
        }
    }

    pub(crate) fn gen_code(&mut self) -> proc_macro2::TokenStream {
        let block_id = core::mem::take(&mut self.block_id);
        let mut child_code = quote::quote!();
        for child in block_id.children.into_iter() {
            child_code.extend(self._gen_code(child));
        }
        child_code
    }

    fn _gen_code(&self, block_id: BlockId) -> proc_macro2::TokenStream {
        let mut body = quote::quote!();
        let block = &self.graph[block_id.id];
        let code = codegen::stmt(block);
        let mut child_code = quote::quote!();
        for child in block_id.children.into_iter() {
            child_code.extend(self._gen_code(child));
        }
        match &block.block_type {
            BlockType::Normal => {
                body.extend(quote::quote!(#code #child_code));
            }
            BlockType::IfCond => {
                body.extend(quote::quote!(if #code #child_code));
            }
            BlockType::ElseIfCond => {
                body.extend(quote::quote!(else if #code #child_code));
            }
            BlockType::IfThen => {
                body.extend(quote::quote!({ #code #child_code }));
            }
            BlockType::IfThenEnd => {
                body.extend(quote::quote!({ #code #child_code };));
            }
            BlockType::IfElseEnd => {
                body.extend(quote::quote!(else { #code #child_code };));
            }
            BlockType::ForInit => {
                body.extend(quote::quote!(for #code #child_code));
            }
            BlockType::ForBody => {
                body.extend(quote::quote!({#code #child_code};));
            }
            BlockType::ForCond => {
                body.extend(quote::quote!(in #code #child_code));
            }
            BlockType::WhileCond => {
                body.extend(quote::quote!(while #code #child_code));
            }
            BlockType::WhileBody => {
                body.extend(quote::quote!({#code #child_code};));
            }
            BlockType::LoopBody => {
                body.extend(quote::quote!(loop {#code #child_code};));
            }
            BlockType::ExprBlock => {
                body.extend(quote::quote!({#code #child_code};));
            }
            BlockType::FnBody => {
                body.extend(quote::quote!({#code #child_code}));
            }
            | BlockType::ExprBlockAssign
            | BlockType::IfAssign
            | BlockType::ForAssign
            | BlockType::WhileAssign
            | BlockType::LoopAssign
            | BlockType::ClosureAssign => {
                body.extend(quote::quote!(let #code #child_code =));
            }
            BlockType::ClosureArgs => {
                body.extend(quote::quote!(|#code #child_code|));
            }
            BlockType::ClosureBody => {
                body.extend(quote::quote!({#code #child_code};));
            }
            BlockType::FnArgs => {
                body.extend(quote::quote!((#code)));
            }
            BlockType::FnVisibility(vis) => {
                body.extend(quote::quote!(#vis));
            }
            BlockType::FnName => {
                body.extend(quote::quote!(fn #code));
            }
            BlockType::FnRet(ret) => {
                body.extend(quote::quote!(#ret));
            }
            BlockType::Generics(generics) => {
                body.extend(quote::quote!(#generics));
            }
            BlockType::Where(where_clause) => {
                body.extend(quote::quote!(#where_clause));
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
    errors: &mut Vec<Error>,
    pat: &mut syn::Pat,
    stacks: &mut HashMap<syn::Ident, Vec<usize>>,
    versions: &mut HashMap<syn::Ident, usize>,
    new_origin_var_map: &mut HashMap<syn::Ident, syn::Ident>
) -> anyhow::Result<()> {
    match pat {
        syn::Pat::Const(const_pat) => {
            errors.push(
                Error::Unsupported(const_pat.span(), "new_name_pat", "const pattern".to_string())
            );
        }
        syn::Pat::Ident(pat_ident) => {
            pat_ident.ident = new_name(
                errors,
                &pat_ident.ident,
                versions,
                stacks,
                new_origin_var_map
            )?;
        }
        syn::Pat::Lit(lit) => {
            errors.push(Error::Unsupported(lit.span(), "new_name_pat", "lit pattern".to_string()));
        }
        syn::Pat::Macro(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "macro pattern".to_string())
            );
        }
        syn::Pat::Or(_) => {
            errors.push(Error::Unsupported(pat.span(), "new_name_pat", "or pattern".to_string()));
        }
        syn::Pat::Paren(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "paren pattern".to_string())
            );
        }
        syn::Pat::Path(_) => {
            errors.push(Error::Unsupported(pat.span(), "new_name_pat", "path pattern".to_string()));
        }
        syn::Pat::Range(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "range pattern".to_string())
            );
        }
        syn::Pat::Reference(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "reference pattern".to_string())
            );
        }
        syn::Pat::Rest(_) => {
            errors.push(Error::Unsupported(pat.span(), "new_name_pat", "rest pattern".to_string()));
        }
        syn::Pat::Slice(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "slice pattern".to_string())
            );
        }
        syn::Pat::Struct(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "struct pattern".to_string())
            );
        }
        syn::Pat::Tuple(tuple) => {
            for pat in tuple.elems.iter_mut() {
                new_name_pat(errors, pat, stacks, versions, new_origin_var_map)?;
            }
        }
        syn::Pat::TupleStruct(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "tuple struct pattern".to_string())
            );
        }
        syn::Pat::Type(pat_type) => {
            new_name_pat(errors, &mut pat_type.pat, stacks, versions, new_origin_var_map)?;
        }
        syn::Pat::Verbatim(_) => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "verbatim pattern".to_string())
            );
        }
        syn::Pat::Wild(_) => {}
        _ => {
            errors.push(
                Error::Unsupported(pat.span(), "new_name_pat", "other pattern".to_string())
            );
        }
    }
    Ok(())
}

fn rename(
    cfg: &mut CFG,
    node: NodeIndex,
    stacks: &mut HashMap<syn::Ident, Vec<usize>>,
    versions: &mut HashMap<syn::Ident, usize>,
    dominators: &Dominators<NodeIndex>
) -> anyhow::Result<()> {
    let mut errors = Vec::new();
    let mut new_origin_var_map = cfg.graph[node].origin_var_map.clone();
    for phi_function in &mut cfg.graph[node].phi_functions {
        for arg in &mut phi_function.args {
            replace_string(arg, stacks);
        }
        let new_name = new_name(
            &mut errors,
            &phi_function.name,
            versions,
            stacks,
            &mut new_origin_var_map
        )?;
        phi_function.name = new_name;
    }
    for stmt in &mut cfg.graph[node].statements {
        match &mut stmt.stmt {
            Stmt::Local(local) => {
                if let Some(init) = &mut local.init {
                    replace_vars(&mut init.expr, stacks, &mut new_origin_var_map);
                }
                new_name_pat(
                    &mut errors,
                    &mut local.pat,
                    stacks,
                    versions,
                    &mut new_origin_var_map
                )?;
            }
            Stmt::Item(item) => {
                replace_vars_item(item, stacks, &mut new_origin_var_map);
            }
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
    cfg.errors.extend(errors);
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
        rename(cfg, succ, stacks, versions, dominators)?;
    }
    for var in &cfg.graph[node].defined_vars {
        stacks.get_mut(var).expect(&format!("rename::phi::stacks: {}", var)).pop();
    }
    Ok(())
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
                    if let Some(var) = expr_path.path.get_ident().cloned() {
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

fn replace_vars_item(
    item: &mut syn::Item,
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
                    if let Some(var) = expr_path.path.get_ident().cloned() {
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
        fn visit_item_mut(&mut self, i: &mut syn::Item) {
            match i {
                syn::Item::Const(item_const) => {
                    if let Some(stack) = self.stacks.get(&item_const.ident) {
                        if let Some(current_cnt) = stack.last() {
                            item_const.ident = syn::Ident::new(
                                &format!("{}{}", item_const.ident, current_cnt),
                                item_const.ident.span()
                            );
                            self.new_origin_var_map.insert(
                                syn::Ident::new(
                                    &format!("{}{}", item_const.ident, current_cnt),
                                    item_const.ident.span()
                                ),
                                item_const.ident.clone()
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }
    syn::visit_mut::visit_item_mut(&mut (VarRenamer { stacks, new_origin_var_map }), item);
}

fn replace_string(var: &mut syn::Ident, stacks: &HashMap<syn::Ident, Vec<usize>>) {
    if let Some(stack) = stacks.get(var) {
        if let Some(current_cnt) = stack.last() {
            *var = syn::Ident::new(&format!("{}{}", var, current_cnt), var.span());
        }
    }
}

fn new_name(
    errors: &mut Vec<Error>,
    var: &syn::Ident,
    versions: &mut HashMap<syn::Ident, usize>,
    stacks: &mut HashMap<syn::Ident, Vec<usize>>,
    new_origin_var_map: &mut HashMap<syn::Ident, syn::Ident>
) -> anyhow::Result<syn::Ident> {
    if let Some(cnt) = versions.get_mut(var) {
        if let Some(stack) = stacks.get_mut(var) {
            stack.push(*cnt);
        }
        let ret = syn::Ident::new(&format!("{}{}", var, cnt), var.span());
        *cnt += 1;
        new_origin_var_map.insert(ret.clone(), var.clone());
        Ok(ret)
    } else {
        errors.push(Error::OriginalVariableNotFound(var.span(), "new_name", var.to_string()));
        Err(
            Error::OriginalVariableNotFound(
                var.span(),
                "new_name",
                var.to_string()
            ).to_anyhow_error()
        )
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
