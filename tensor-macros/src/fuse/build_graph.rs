use std::{ collections::{ HashMap, HashSet }, rc::Rc };

use petgraph::prelude::StableGraph;
use proc_macro2::Span;
use quote::ToTokens;
use syn::spanned::Spanned;

use super::{ node::{ Binary, Node, Unary }, ty_infer::Type, variable_collector::VariableCollector };

pub(crate) struct Graph {
    pub(crate) nodes: Vec<(Node, i64, usize)>,
    pub(crate) inputs: HashMap<(Node, i64, usize), Type>,
    pub(crate) type_table: Rc<HashMap<syn::Ident, Type>>,
    pub(crate) variables: HashSet<syn::Ident>,
    pub(crate) current_var: syn::Ident,
    pub(crate) tmp_var_version: usize,
    pub(crate) current_assignment: Option<syn::Ident>,
    pub(crate) current_idx: usize,
    pub(crate) current_block: usize,
    pub(crate) extra_temps: Vec<syn::Ident>,
}

impl Graph {
    pub fn new(type_table: Rc<HashMap<syn::Ident, Type>>, current_block: usize) -> Self {
        Self {
            type_table,
            nodes: vec![],
            inputs: HashMap::new(),
            variables: HashSet::new(),
            current_var: syn::Ident::new("__out", Span::call_site()),
            tmp_var_version: 0,
            current_assignment: None,
            current_idx: 0,
            current_block,
            extra_temps: vec![],
        }
    }

    pub(crate) fn to_petgraph(&self) -> StableGraph<&(Node, i64, usize), ()> {
        let mut graph = StableGraph::new();
        let mut node_index_map = HashMap::new();

        // 将节点添加到 petgraph
        for node in &self.nodes {
            let index = graph.add_node(node);
            node_index_map.insert(node, index);
        }

        for (input, ty) in &self.inputs {
            if ty.is_tensor() {
                let index = graph.add_node(input);
                node_index_map.insert(input, index);
            }
        }

        // 添加边（假设 Node 有一个 `dependencies` 字段存储依赖节点的索引）
        for node in &self.nodes {
            match &node.0 {
                Node::Unary(unary) => {
                    if
                        let Some(tuple) = self.nodes.iter().find(|(node, _, _)| {
                            match node {
                                Node::Unary(node) => node.output == unary.operand,
                                Node::Binary(node) => { node.output == unary.operand }
                                Node::Input(_) => false,
                            }
                        })
                    {
                        graph.add_edge(node_index_map[tuple], node_index_map[node], ());
                    }
                }
                Node::Binary(binary) => {
                    if
                        let Some(tuple) = self.nodes.iter().find(|(node, _, _)| {
                            match node {
                                Node::Unary(node) =>
                                    node.output == binary.left || node.output == binary.right,
                                Node::Binary(node) =>
                                    node.output == binary.left || node.output == binary.right,
                                Node::Input(_) => false,
                            }
                        })
                    {
                        graph.add_edge(node_index_map[tuple], node_index_map[node], ());
                    }
                }
                Node::Input(_) => unreachable!(),
            }
        }

        for (inp, ty) in &self.inputs {
            if ty.is_tensor() {
                if let Node::Input(input) = &inp.0 {
                    if let Some(index) = node_index_map.get(inp) {
                        let node_indices = graph.node_indices().collect::<Vec<_>>();
                        for node_idx in node_indices {
                            let (node, _, _) = graph
                                .node_weight(node_idx)
                                .expect("node weight not found");
                            match node {
                                Node::Unary(unary) => {
                                    if &unary.operand == input {
                                        graph.add_edge(*index, node_idx, ());
                                    }
                                }
                                Node::Binary(binary) => {
                                    if &binary.left == input || &binary.right == input {
                                        graph.add_edge(*index, node_idx, ());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        graph
    }

    fn push_input_node_if_not_exist_expr(&mut self, expr: &syn::Expr) {
        match expr {
            syn::Expr::Array(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Array"),
            syn::Expr::Assign(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Assign"),
            syn::Expr::Async(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Async"),
            syn::Expr::Await(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Await"),
            syn::Expr::Binary(binary) => {
                self.push_input_node_if_not_exist_expr(&binary.left);
                self.push_input_node_if_not_exist_expr(&binary.right);
            }
            syn::Expr::Block(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Block"),
            syn::Expr::Break(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Break"),
            syn::Expr::Call(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Call"),
            syn::Expr::Cast(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Cast"),
            syn::Expr::Closure(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::Closure"
                ),
            syn::Expr::Const(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Const"),
            syn::Expr::Continue(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::Continue"
                ),
            syn::Expr::Field(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Field"),
            syn::Expr::ForLoop(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::ForLoop"
                ),
            syn::Expr::Group(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Group"),
            syn::Expr::If(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::If"),
            syn::Expr::Index(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Index"),
            syn::Expr::Infer(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Infer"),
            syn::Expr::Let(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Let"),
            syn::Expr::Lit(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Lit"),
            syn::Expr::Loop(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Loop"),
            syn::Expr::Macro(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Macro"),
            syn::Expr::Match(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Match"),
            syn::Expr::MethodCall(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::MethodCall"
                ),
            syn::Expr::Paren(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Paren"),
            syn::Expr::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    if self.variables.contains(&ident) {
                        return;
                    }
                    self.inputs.insert(
                        (
                            Node::Input(ident.clone()),
                            -1,
                            self.current_block,
                        ),
                        self.type_table[&ident]
                    );
                    self.variables.insert(ident.clone());
                }
            }
            syn::Expr::Range(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Range"),
            syn::Expr::RawAddr(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::RawAddr"
                ),
            syn::Expr::Reference(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::Reference"
                ),
            syn::Expr::Repeat(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Repeat"),
            syn::Expr::Return(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Return"),
            syn::Expr::Struct(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Struct"),
            syn::Expr::Try(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Try"),
            syn::Expr::TryBlock(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::TryBlock"
                ),
            syn::Expr::Tuple(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Tuple"),
            syn::Expr::Unary(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Unary"),
            syn::Expr::Unsafe(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Unsafe"),
            syn::Expr::Verbatim(_) =>
                unimplemented!(
                    "build_graph::push_input_node_if_not_exist_expr::syn::Expr::Verbatim"
                ),
            syn::Expr::While(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::While"),
            syn::Expr::Yield(_) =>
                unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Yield"),
            _ => unimplemented!("build_graph::push_input_node_if_not_exist_expr::syn::Expr::Other"),
        }
    }
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph").field("inputs", &self.inputs).field("nodes", &self.nodes).finish()
    }
}

impl<'ast> syn::visit::Visit<'ast> for Graph {
    fn visit_abi(&mut self, _: &'ast syn::Abi) {
        unimplemented!("graph::visit_abi");
    }

    fn visit_angle_bracketed_generic_arguments(
        &mut self,
        _: &'ast syn::AngleBracketedGenericArguments
    ) {
        unimplemented!("graph::visit_angle_bracketed_generic_arguments");
    }

    fn visit_arm(&mut self, _: &'ast syn::Arm) {
        unimplemented!("graph::visit_arm");
    }

    fn visit_assoc_const(&mut self, _: &'ast syn::AssocConst) {
        unimplemented!("graph::visit_assoc_const");
    }

    fn visit_assoc_type(&mut self, _: &'ast syn::AssocType) {
        unimplemented!("graph::visit_assoc_type");
    }

    fn visit_bare_fn_arg(&mut self, _: &'ast syn::BareFnArg) {
        unimplemented!("graph::visit_bare_fn_arg");
    }

    fn visit_bare_variadic(&mut self, _: &'ast syn::BareVariadic) {
        unimplemented!("graph::visit_bare_variadic");
    }

    fn visit_bin_op(&mut self, _: &'ast syn::BinOp) {
        unimplemented!("graph::visit_bin_op");
    }

    fn visit_bound_lifetimes(&mut self, _: &'ast syn::BoundLifetimes) {
        unimplemented!("graph::visit_bound_lifetimes");
    }

    fn visit_captured_param(&mut self, _: &'ast syn::CapturedParam) {
        unimplemented!("graph::visit_captured_param");
    }

    fn visit_const_param(&mut self, _: &'ast syn::ConstParam) {
        unimplemented!("graph::visit_const_param");
    }

    fn visit_constraint(&mut self, _: &'ast syn::Constraint) {
        unimplemented!("graph::visit_constraint");
    }

    fn visit_data(&mut self, _: &'ast syn::Data) {
        unimplemented!("graph::visit_data");
    }

    fn visit_data_enum(&mut self, _: &'ast syn::DataEnum) {
        unimplemented!("graph::visit_data_enum");
    }

    fn visit_data_struct(&mut self, _: &'ast syn::DataStruct) {
        unimplemented!("graph::visit_data_struct");
    }

    fn visit_data_union(&mut self, _: &'ast syn::DataUnion) {
        unimplemented!("graph::visit_data_union");
    }

    fn visit_derive_input(&mut self, _: &'ast syn::DeriveInput) {
        unimplemented!("graph::visit_derive_input");
    }

    fn visit_expr_array(&mut self, _: &'ast syn::ExprArray) {
        unimplemented!("graph::visit_expr_array");
    }

    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        for it in &assign.attrs {
            self.visit_attribute(it);
        }
        match assign.left.as_ref() {
            syn::Expr::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    self.visit_ident(ident);
                    self.current_assignment = Some(ident.clone());
                    self.visit_expr(&assign.right);
                    self.variables.insert(ident.clone());
                }
            }
            _ => {
                unimplemented!("graph::visit_expr_assign::left");
            }
        }
    }

    fn visit_expr_async(&mut self, _: &'ast syn::ExprAsync) {
        unimplemented!("graph::visit_expr_async");
    }

    fn visit_expr_await(&mut self, _: &'ast syn::ExprAwait) {
        unimplemented!("graph::visit_expr_await");
    }

    fn visit_expr_block(&mut self, _: &'ast syn::ExprBlock) {
        unimplemented!("graph::visit_expr_block");
    }

    fn visit_expr_cast(&mut self, _: &'ast syn::ExprCast) {
        unimplemented!("graph::visit_expr_cast");
    }

    fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {
        unimplemented!("graph::visit_expr_closure");
    }

    fn visit_expr_const(&mut self, _: &'ast syn::ExprConst) {
        unimplemented!("graph::visit_expr_const");
    }

    fn visit_expr_field(&mut self, _: &'ast syn::ExprField) {
        unimplemented!("graph::visit_expr_field");
    }

    fn visit_expr_for_loop(&mut self, _: &'ast syn::ExprForLoop) {
        unimplemented!("graph::visit_expr_for_loop");
    }

    fn visit_expr_group(&mut self, _: &'ast syn::ExprGroup) {
        unimplemented!("graph::visit_expr_group");
    }

    fn visit_expr_index(&mut self, _: &'ast syn::ExprIndex) {
        unimplemented!("graph::visit_expr_index");
    }

    fn visit_expr_infer(&mut self, _: &'ast syn::ExprInfer) {
        unimplemented!("graph::visit_expr_infer");
    }

    fn visit_expr_let(&mut self, _: &'ast syn::ExprLet) {
        unimplemented!("graph::visit_expr_let");
    }

    fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {
        unimplemented!("graph::visit_expr_loop");
    }

    fn visit_expr_macro(&mut self, _: &'ast syn::ExprMacro) {
        unimplemented!("graph::visit_expr_macro");
    }

    fn visit_expr_match(&mut self, _: &'ast syn::ExprMatch) {
        unimplemented!("graph::visit_expr_match");
    }

    fn visit_expr_paren(&mut self, _: &'ast syn::ExprParen) {
        unimplemented!("graph::visit_expr_paren");
    }

    fn visit_expr_raw_addr(&mut self, _: &'ast syn::ExprRawAddr) {
        unimplemented!("graph::visit_expr_raw_addr");
    }

    fn visit_expr_repeat(&mut self, _: &'ast syn::ExprRepeat) {
        unimplemented!("graph::visit_expr_repeat");
    }

    fn visit_expr_return(&mut self, _: &'ast syn::ExprReturn) {
        unimplemented!("graph::visit_expr_return");
    }

    fn visit_expr_struct(&mut self, _: &'ast syn::ExprStruct) {
        unimplemented!("graph::visit_expr_struct");
    }

    fn visit_expr_try_block(&mut self, _: &'ast syn::ExprTryBlock) {
        unimplemented!("graph::visit_expr_try_block");
    }

    fn visit_expr_unary(&mut self, _: &'ast syn::ExprUnary) {
        unimplemented!("graph::visit_expr_unary");
    }

    fn visit_expr_unsafe(&mut self, _: &'ast syn::ExprUnsafe) {
        unimplemented!("graph::visit_expr_unsafe");
    }

    fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {
        unimplemented!("graph::visit_expr_while");
    }

    fn visit_expr_yield(&mut self, _: &'ast syn::ExprYield) {
        unimplemented!("graph::visit_expr_yield");
    }

    fn visit_field(&mut self, _: &'ast syn::Field) {
        unimplemented!("graph::visit_field");
    }

    fn visit_field_mutability(&mut self, _: &'ast syn::FieldMutability) {
        unimplemented!("graph::visit_field_mutability");
    }

    fn visit_field_pat(&mut self, _: &'ast syn::FieldPat) {
        unimplemented!("graph::visit_field_pat");
    }

    fn visit_field_value(&mut self, _: &'ast syn::FieldValue) {
        unimplemented!("graph::visit_field_value");
    }

    fn visit_fields(&mut self, _: &'ast syn::Fields) {
        unimplemented!("graph::visit_fields");
    }

    fn visit_fields_named(&mut self, _: &'ast syn::FieldsNamed) {
        unimplemented!("graph::visit_fields_named");
    }

    fn visit_fields_unnamed(&mut self, _: &'ast syn::FieldsUnnamed) {
        unimplemented!("graph::visit_fields_unnamed");
    }

    fn visit_file(&mut self, _: &'ast syn::File) {
        unimplemented!("graph::visit_file");
    }

    fn visit_foreign_item(&mut self, _: &'ast syn::ForeignItem) {
        unimplemented!("graph::visit_foreign_item");
    }

    fn visit_foreign_item_fn(&mut self, _: &'ast syn::ForeignItemFn) {
        unimplemented!("graph::visit_foreign_item_fn");
    }

    fn visit_foreign_item_macro(&mut self, _: &'ast syn::ForeignItemMacro) {
        unimplemented!("graph::visit_foreign_item_macro");
    }

    fn visit_foreign_item_static(&mut self, _: &'ast syn::ForeignItemStatic) {
        unimplemented!("graph::visit_foreign_item_static");
    }

    fn visit_foreign_item_type(&mut self, _: &'ast syn::ForeignItemType) {
        unimplemented!("graph::visit_foreign_item_type");
    }

    fn visit_generic_argument(&mut self, _: &'ast syn::GenericArgument) {
        unimplemented!("graph::visit_generic_argument");
    }

    fn visit_generic_param(&mut self, _: &'ast syn::GenericParam) {
        unimplemented!("graph::visit_generic_param");
    }

    fn visit_impl_item(&mut self, _: &'ast syn::ImplItem) {
        unimplemented!("graph::visit_impl_item");
    }

    fn visit_impl_item_const(&mut self, _: &'ast syn::ImplItemConst) {
        unimplemented!("graph::visit_impl_item_const");
    }

    fn visit_impl_item_fn(&mut self, _: &'ast syn::ImplItemFn) {
        unimplemented!("graph::visit_impl_item_fn");
    }

    fn visit_impl_item_macro(&mut self, _: &'ast syn::ImplItemMacro) {
        unimplemented!("graph::visit_impl_item_macro");
    }

    fn visit_impl_item_type(&mut self, _: &'ast syn::ImplItemType) {
        unimplemented!("graph::visit_impl_item_type");
    }

    fn visit_impl_restriction(&mut self, _: &'ast syn::ImplRestriction) {
        unimplemented!("graph::visit_impl_restriction");
    }

    fn visit_index(&mut self, _: &'ast syn::Index) {
        unimplemented!("graph::visit_index");
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {
        unimplemented!("graph::visit_item");
    }

    fn visit_item_const(&mut self, _: &'ast syn::ItemConst) {
        unimplemented!("graph::visit_item_const");
    }

    fn visit_item_enum(&mut self, _: &'ast syn::ItemEnum) {
        unimplemented!("graph::visit_item_enum");
    }

    fn visit_item_extern_crate(&mut self, _: &'ast syn::ItemExternCrate) {
        unimplemented!("graph::visit_item_extern_crate");
    }

    fn visit_item_fn(&mut self, _: &'ast syn::ItemFn) {
        unimplemented!("graph::visit_item_fn");
    }

    fn visit_item_foreign_mod(&mut self, _: &'ast syn::ItemForeignMod) {
        unimplemented!("graph::visit_item_foreign_mod");
    }

    fn visit_item_impl(&mut self, _: &'ast syn::ItemImpl) {
        unimplemented!("graph::visit_item_impl");
    }

    fn visit_item_macro(&mut self, _: &'ast syn::ItemMacro) {
        unimplemented!("graph::visit_item_macro");
    }

    fn visit_item_mod(&mut self, _: &'ast syn::ItemMod) {
        unimplemented!("graph::visit_item_mod");
    }

    fn visit_item_static(&mut self, _: &'ast syn::ItemStatic) {
        unimplemented!("graph::visit_item_static");
    }

    fn visit_item_struct(&mut self, _: &'ast syn::ItemStruct) {
        unimplemented!("graph::visit_item_struct");
    }

    fn visit_item_trait(&mut self, _: &'ast syn::ItemTrait) {
        unimplemented!("graph::visit_item_trait");
    }

    fn visit_item_trait_alias(&mut self, _: &'ast syn::ItemTraitAlias) {
        unimplemented!("graph::visit_item_trait_alias");
    }

    fn visit_item_type(&mut self, _: &'ast syn::ItemType) {
        unimplemented!("graph::visit_item_type");
    }

    fn visit_item_union(&mut self, _: &'ast syn::ItemUnion) {
        unimplemented!("graph::visit_item_union");
    }

    fn visit_item_use(&mut self, _: &'ast syn::ItemUse) {
        unimplemented!("graph::visit_item_use");
    }

    fn visit_label(&mut self, _: &'ast syn::Label) {
        unimplemented!("graph::visit_label");
    }

    fn visit_member(&mut self, _: &'ast syn::Member) {
        unimplemented!("graph::visit_member");
    }

    fn visit_meta_list(&mut self, _: &'ast syn::MetaList) {
        unimplemented!("graph::visit_meta_list");
    }

    fn visit_meta_name_value(&mut self, _: &'ast syn::MetaNameValue) {
        unimplemented!("graph::visit_meta_name_value");
    }

    fn visit_parenthesized_generic_arguments(
        &mut self,
        _: &'ast syn::ParenthesizedGenericArguments
    ) {
        unimplemented!("graph::visit_parenthesized_generic_arguments");
    }

    fn visit_pat_or(&mut self, _: &'ast syn::PatOr) {
        unimplemented!("graph::visit_pat_or");
    }

    fn visit_pat_paren(&mut self, _: &'ast syn::PatParen) {
        unimplemented!("graph::visit_pat_paren");
    }

    fn visit_pat_reference(&mut self, _: &'ast syn::PatReference) {
        unimplemented!("graph::visit_pat_reference");
    }

    fn visit_pat_rest(&mut self, _: &'ast syn::PatRest) {
        unimplemented!("graph::visit_pat_rest");
    }

    fn visit_pat_slice(&mut self, _: &'ast syn::PatSlice) {
        unimplemented!("graph::visit_pat_slice");
    }

    fn visit_pat_struct(&mut self, _: &'ast syn::PatStruct) {
        unimplemented!("graph::visit_pat_struct");
    }

    fn visit_pat_tuple(&mut self, _: &'ast syn::PatTuple) {
        unimplemented!("graph::visit_pat_tuple");
    }

    fn visit_pat_tuple_struct(&mut self, _: &'ast syn::PatTupleStruct) {
        unimplemented!("graph::visit_pat_tuple_struct");
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if let Some(ident) = path.get_ident() {
            self.visit_ident(ident);
        }
    }

    fn visit_path_arguments(&mut self, _: &'ast syn::PathArguments) {
        unimplemented!("graph::visit_path_arguments");
    }

    fn visit_path_segment(&mut self, segment: &'ast syn::PathSegment) {
        unimplemented!("graph::visit_path_segment::{}", segment.to_token_stream().to_string());
    }

    fn visit_pointer_mutability(&mut self, _: &'ast syn::PointerMutability) {
        unimplemented!("graph::visit_pointer_mutability");
    }

    fn visit_precise_capture(&mut self, _: &'ast syn::PreciseCapture) {
        unimplemented!("graph::visit_precise_capture");
    }

    fn visit_predicate_lifetime(&mut self, _: &'ast syn::PredicateLifetime) {
        unimplemented!("graph::visit_predicate_lifetime");
    }

    fn visit_predicate_type(&mut self, _: &'ast syn::PredicateType) {
        unimplemented!("graph::visit_predicate_type");
    }

    fn visit_qself(&mut self, _: &'ast syn::QSelf) {
        unimplemented!("graph::visit_qself");
    }

    fn visit_receiver(&mut self, _: &'ast syn::Receiver) {
        unimplemented!("graph::visit_receiver");
    }

    fn visit_static_mutability(&mut self, _: &'ast syn::StaticMutability) {
        unimplemented!("graph::visit_static_mutability");
    }

    fn visit_trait_bound(&mut self, _: &'ast syn::TraitBound) {
        unimplemented!("graph::visit_trait_bound");
    }

    fn visit_trait_bound_modifier(&mut self, _: &'ast syn::TraitBoundModifier) {
        unimplemented!("graph::visit_trait_bound_modifier");
    }

    fn visit_trait_item(&mut self, _: &'ast syn::TraitItem) {
        unimplemented!("graph::visit_trait_item");
    }

    fn visit_trait_item_const(&mut self, _: &'ast syn::TraitItemConst) {
        unimplemented!("graph::visit_trait_item_const");
    }

    fn visit_trait_item_fn(&mut self, _: &'ast syn::TraitItemFn) {
        unimplemented!("graph::visit_trait_item_fn");
    }

    fn visit_trait_item_macro(&mut self, _: &'ast syn::TraitItemMacro) {
        unimplemented!("graph::visit_trait_item_macro");
    }

    fn visit_trait_item_type(&mut self, _: &'ast syn::TraitItemType) {
        unimplemented!("graph::visit_trait_item_type");
    }

    fn visit_type(&mut self, _: &'ast syn::Type) {
        unimplemented!("graph::visit_type");
    }

    fn visit_type_array(&mut self, _: &'ast syn::TypeArray) {
        unimplemented!("graph::visit_type_array");
    }

    fn visit_type_bare_fn(&mut self, _: &'ast syn::TypeBareFn) {
        unimplemented!("graph::visit_type_bare_fn");
    }

    fn visit_type_group(&mut self, _: &'ast syn::TypeGroup) {
        unimplemented!("graph::visit_type_group");
    }

    fn visit_type_impl_trait(&mut self, _: &'ast syn::TypeImplTrait) {
        unimplemented!("graph::visit_type_impl_trait");
    }

    fn visit_type_infer(&mut self, _: &'ast syn::TypeInfer) {
        unimplemented!("graph::visit_type_infer");
    }

    fn visit_type_macro(&mut self, _: &'ast syn::TypeMacro) {
        unimplemented!("graph::visit_type_macro");
    }

    fn visit_type_never(&mut self, _: &'ast syn::TypeNever) {
        unimplemented!("graph::visit_type_never");
    }

    fn visit_type_param(&mut self, _: &'ast syn::TypeParam) {
        unimplemented!("graph::visit_type_param");
    }

    fn visit_type_param_bound(&mut self, _: &'ast syn::TypeParamBound) {
        unimplemented!("graph::visit_type_param_bound");
    }

    fn visit_type_paren(&mut self, _: &'ast syn::TypeParen) {
        unimplemented!("graph::visit_type_paren");
    }

    fn visit_type_path(&mut self, _: &'ast syn::TypePath) {
        unimplemented!("graph::visit_type_path");
    }

    fn visit_type_ptr(&mut self, _: &'ast syn::TypePtr) {
        unimplemented!("graph::visit_type_ptr");
    }

    fn visit_type_reference(&mut self, _: &'ast syn::TypeReference) {
        unimplemented!("graph::visit_type_reference");
    }

    fn visit_type_slice(&mut self, _: &'ast syn::TypeSlice) {
        unimplemented!("graph::visit_type_slice");
    }

    fn visit_type_trait_object(&mut self, _: &'ast syn::TypeTraitObject) {
        unimplemented!("graph::visit_type_trait_object");
    }

    fn visit_type_tuple(&mut self, _: &'ast syn::TypeTuple) {
        unimplemented!("graph::visit_type_tuple");
    }

    fn visit_un_op(&mut self, _: &'ast syn::UnOp) {
        unimplemented!("graph::visit_un_op");
    }

    fn visit_use_glob(&mut self, _: &'ast syn::UseGlob) {
        unimplemented!("graph::visit_use_glob");
    }

    fn visit_use_group(&mut self, _: &'ast syn::UseGroup) {
        unimplemented!("graph::visit_use_group");
    }

    fn visit_use_name(&mut self, _: &'ast syn::UseName) {
        unimplemented!("graph::visit_use_name");
    }

    fn visit_use_path(&mut self, _: &'ast syn::UsePath) {
        unimplemented!("graph::visit_use_path");
    }

    fn visit_use_rename(&mut self, _: &'ast syn::UseRename) {
        unimplemented!("graph::visit_use_rename");
    }

    fn visit_use_tree(&mut self, _: &'ast syn::UseTree) {
        unimplemented!("graph::visit_use_tree");
    }

    fn visit_variadic(&mut self, _: &'ast syn::Variadic) {
        unimplemented!("graph::visit_variadic");
    }

    fn visit_variant(&mut self, _: &'ast syn::Variant) {
        unimplemented!("graph::visit_variant");
    }

    fn visit_vis_restricted(&mut self, _: &'ast syn::VisRestricted) {
        unimplemented!("graph::visit_vis_restricted");
    }

    fn visit_where_clause(&mut self, _: &'ast syn::WhereClause) {
        unimplemented!("graph::visit_where_clause");
    }

    fn visit_where_predicate(&mut self, _: &'ast syn::WherePredicate) {
        unimplemented!("graph::visit_where_predicate");
    }
    fn visit_block(&mut self, _: &'ast syn::Block) {
        unimplemented!("graph::visit_block");
    }

    fn visit_pat_type(&mut self, node: &'ast syn::PatType) {
        let mut collector = VariableCollector::new();
        collector.visit_pat(&node.pat);
        self.variables.extend(collector.vars);
    }
    fn visit_local(&mut self, local: &'ast syn::Local) {
        for it in &local.attrs {
            self.visit_attribute(it);
        }
        match &local.pat {
            syn::Pat::Ident(pat_ident) => {
                self.visit_pat_ident(pat_ident);
                self.current_assignment = Some(pat_ident.ident.clone());
                if let Some(it) = &local.init {
                    self.visit_local_init(it);
                    self.variables.insert(pat_ident.ident.clone());
                }
            }
            _ => {
                syn::visit::visit_pat(self, &local.pat);
                if let Some(it) = &local.init {
                    self.visit_local_init(it);
                }
            }
        }
    }
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let current_assignment = self.current_assignment.clone();
        self.visit_expr(&node.receiver);

        let is_assignment = current_assignment.is_some();
        let out = if let Some(current_assignment) = current_assignment {
            self.current_assignment = None;
            self.current_var = current_assignment.clone();
            current_assignment
        } else {
            let out = proc_macro2::Ident::new(
                &format!("__out_{}", self.tmp_var_version),
                node.span()
            );
            self.variables.insert(out.clone());
            self.extra_temps.push(out.clone());
            self.current_var = out.clone();
            self.tmp_var_version += 1;
            out
        };
        let args = match node.method.to_string().as_str() {
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "relu" => {
                vec![]
            }
            "selu" => {
                node.args
                    .iter()
                    .map(|arg| arg.clone())
                    .collect()
            }
            _ =>
                unimplemented!(
                    "_visitor::visit_expr_method_call::{}",
                    node.method.to_string().as_str()
                ),
        };
        for arg in args.iter() {
            self.push_input_node_if_not_exist_expr(arg);
        }
        self.push_input_node_if_not_exist_expr(&node.receiver);
        let operand = node.receiver.to_token_stream().to_string();
        let method = Node::Unary(Unary {
            method: node.method.clone(),
            operand: proc_macro2::Ident::new(&operand, node.span()),
            args,
            output: out.clone(),
        });
        self.nodes.push((
            method,
            if is_assignment { self.current_idx as i64 } else { -1 },
            self.current_block,
        ));
    }

    fn visit_ident(&mut self, i: &'ast proc_macro2::Ident) {
        self.current_var = i.clone();
    }
    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        if i.path.get_ident().is_some() {
            self.visit_ident(&i.path.segments[0].ident);
        } else {
            syn::visit::visit_expr_path(self, i);
        }
    }
    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        let left_ty = handle_expr_type(&node.left, &self.type_table);
        let right_ty = handle_expr_type(&node.right, &self.type_table);
        if left_ty != Type::Tensor && right_ty != Type::Tensor {
            return;
        }
        let current_assignment = self.current_assignment.clone();
        self.current_assignment = None;
        self.visit_expr(&node.left);
        let left_var = self.current_var.clone();
        if !self.variables.contains(&self.current_var) {
            self.variables.insert(left_var.clone());
            let node = Node::Input(left_var.clone());
            if !self.inputs.contains_key(&(node.clone(), -1, self.current_block)) {
                self.inputs.insert(
                    (node, -1, self.current_block),
                    self.type_table[&left_var].clone()
                );
            }
        }
        self.visit_expr(&node.right);
        let right_var = self.current_var.clone();
        if !self.variables.contains(&right_var) {
            self.variables.insert(right_var.clone());
            let node = Node::Input(right_var.clone());
            if !self.inputs.contains_key(&(node.clone(), -1, self.current_block)) {
                self.inputs.insert(
                    (node, -1, self.current_block),
                    self.type_table[&right_var].clone()
                );
            }
        }
        let method = match node.op {
            syn::BinOp::Add(_) => "add",
            syn::BinOp::Sub(_) => "sub",
            syn::BinOp::Mul(_) => "mul",
            syn::BinOp::Div(_) => "div",
            syn::BinOp::Rem(_) => "rem",
            syn::BinOp::And(_) => "and",
            syn::BinOp::Or(_) => "or",
            syn::BinOp::BitXor(_) => "bitxor",
            syn::BinOp::BitAnd(_) => "bitand",
            syn::BinOp::BitOr(_) => "bitor",
            syn::BinOp::Shl(_) => "shl",
            syn::BinOp::Shr(_) => "shr",
            syn::BinOp::Eq(_) => "eq",
            syn::BinOp::Lt(_) => "lt",
            syn::BinOp::Le(_) => "le",
            syn::BinOp::Ne(_) => "ne",
            syn::BinOp::Ge(_) => "ge",
            syn::BinOp::Gt(_) => "gt",
            syn::BinOp::AddAssign(_) => "add_assign",
            syn::BinOp::SubAssign(_) => "sub_assign",
            syn::BinOp::MulAssign(_) => "mul_assign",
            syn::BinOp::DivAssign(_) => "div_assign",
            syn::BinOp::RemAssign(_) => "rem_assign",
            syn::BinOp::BitXorAssign(_) => "bitxor_assign",
            syn::BinOp::BitAndAssign(_) => "bitand_assign",
            syn::BinOp::BitOrAssign(_) => "bitor_assign",
            syn::BinOp::ShlAssign(_) => "shl_assign",
            syn::BinOp::ShrAssign(_) => "shr_assign",
            _ => todo!(),
        };

        let is_assignment = current_assignment.is_some();
        let out = if let Some(current_assignment) = current_assignment {
            self.current_assignment = None;
            self.current_var = current_assignment.clone();
            current_assignment
        } else {
            let out = proc_macro2::Ident::new(
                &format!("__out_{}", self.tmp_var_version),
                node.span()
            );
            self.variables.insert(out.clone());
            self.extra_temps.push(out.clone());
            self.current_var = out.clone();
            self.tmp_var_version += 1;
            out
        };
        self.nodes.push((
            Node::Binary(Binary {
                method: proc_macro2::Ident::new(method, node.span()),
                left: left_var,
                right: right_var,
                output: out.clone(),
            }),
            if is_assignment { self.current_idx as i64 } else { -1 },
            self.current_block,
        ));
    }
}

fn handle_expr_type<'ast>(node: &'ast syn::Expr, type_table: &HashMap<syn::Ident, Type>) -> Type {
    match node {
        syn::Expr::Binary(expr_binary) => {
            let left_ty = handle_expr_type(&expr_binary.left, type_table);
            let right_ty = handle_expr_type(&expr_binary.right, type_table);
            if left_ty == Type::Tensor || right_ty == Type::Tensor {
                Type::Tensor
            } else {
                Type::Unknown
            }
        }
        syn::Expr::Call(_) => unimplemented!("build_graph::handle_expr_type::Call"),
        syn::Expr::Cast(_) => unimplemented!("build_graph::handle_expr_type::Cast"),
        syn::Expr::Closure(_) => unimplemented!("build_graph::handle_expr_type::Closure"),
        syn::Expr::Const(_) => unimplemented!("build_graph::handle_expr_type::Const"),
        syn::Expr::Continue(_) => unimplemented!("build_graph::handle_expr_type::Continue"),
        syn::Expr::Field(_) => unimplemented!("build_graph::handle_expr_type::Field"),
        syn::Expr::Group(_) => unimplemented!("build_graph::handle_expr_type::Group"),
        syn::Expr::Index(_) => unimplemented!("build_graph::handle_expr_type::Index"),
        syn::Expr::Infer(_) => unimplemented!("build_graph::handle_expr_type::Infer"),
        syn::Expr::Let(_) => unimplemented!("build_graph::handle_expr_type::Let"),
        syn::Expr::Lit(_) => Type::Scalar,
        syn::Expr::Macro(_) => unimplemented!("build_graph::handle_expr_type::Macro"),
        syn::Expr::Match(_) => unimplemented!("build_graph::handle_expr_type::Match"),
        syn::Expr::MethodCall(_) => unimplemented!("build_graph::handle_expr_type::MethodCall"),
        syn::Expr::Paren(_) => unimplemented!("build_graph::handle_expr_type::Paren"),
        syn::Expr::Path(expr_path) => {
            if let Some(ident) = expr_path.path.get_ident() {
                type_table.get(ident).unwrap_or(&Type::Unknown).clone()
            } else {
                Type::Unknown
            }
        }
        syn::Expr::Range(_) => unimplemented!("build_graph::handle_expr_type::Range"),
        syn::Expr::RawAddr(_) => unimplemented!("build_graph::handle_expr_type::RawAddr"),
        syn::Expr::Reference(reference) => { handle_expr_type(&reference.expr, type_table) }
        syn::Expr::Repeat(_) => unimplemented!("build_graph::handle_expr_type::Repeat"),
        syn::Expr::Return(_) => unimplemented!("build_graph::handle_expr_type::Return"),
        syn::Expr::Struct(_) => unimplemented!("build_graph::handle_expr_type::Struct"),
        syn::Expr::Try(_) => unimplemented!("build_graph::handle_expr_type::Try"),
        syn::Expr::TryBlock(_) => unimplemented!("build_graph::handle_expr_type::TryBlock"),
        syn::Expr::Tuple(_) => unimplemented!("build_graph::handle_expr_type::Tuple"),
        syn::Expr::Unary(_) => unimplemented!("build_graph::handle_expr_type::Unary"),
        syn::Expr::Unsafe(_) => unimplemented!("build_graph::handle_expr_type::Unsafe"),
        syn::Expr::Verbatim(_) => unimplemented!("build_graph::handle_expr_type::Verbatim"),
        syn::Expr::Yield(_) => unimplemented!("build_graph::handle_expr_type::Yield"),
        _ => Type::Unknown,
    }
}
