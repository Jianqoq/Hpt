use std::{ collections::{ HashMap, HashSet }, rc::Rc };

use petgraph::{ graph::NodeIndex, prelude::StableGraph };
use quote::ToTokens;
use syn::spanned::Spanned;

use super::{
    errors::Error,
    kernel_type::KernelType,
    node::{ Binary, Node, Operand, Unary },
    operator_lists::{ BINARY_OPERATORS, OPAQUE_BINARY_OPERATORS, UNARY_OPERATORS },
    ty_infer::Type,
    variable_collector::VariableCollector,
};

#[derive(Clone)]
pub(crate) struct CmpNode {
    pub(crate) kernel_type: KernelType,
    pub(crate) args: Vec<NodeIndex>,
    pub(crate) outputs: Vec<NodeIndex>,
    pub(crate) args_ident: Vec<Operand>,
    pub(crate) outputs_ident: Vec<Operand>,
    pub(crate) stmt_idx: i64,
    pub(crate) block_idx: usize,
    pub(crate) id: NodeIndex,
    pub(crate) ident: Operand,
    pub(crate) method: Option<syn::Ident>,
}

impl ToTokens for CmpNode {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        if let Some(method) = &self.method {
            let method = proc_macro2::Ident::new(
                &format!("_{}", method.to_string()),
                method.span()
            );
            match self.args_ident.len() {
                1 => {
                    let left = &self.args_ident[0];
                    let output = &self.ident;
                    tokens.extend(
                        quote::quote!(
                        let #output = #left.#method();
                    )
                    );
                }
                2 => {
                    let left = &self.args_ident[0];
                    let right = &self.args_ident[1];
                    let output = &self.ident;
                    tokens.extend(
                        quote::quote!(
                         let #output = #left.#method(#right);
                    )
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

impl std::fmt::Debug for CmpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(method) = &self.method {
            write!(
                f,
                "{}: {} = {}({})",
                self.ident.to_string(),
                self.id.index(),
                method.to_string(),
                self.args_ident
                    .iter()
                    .zip(self.args.iter())
                    .map(|(x, y)| format!("{}: {}", x.to_string(), y.index()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            write!(f, "{}: {}", self.ident.to_string(), self.id.index())
        }
    }
}

pub(crate) struct Graph {
    pub(crate) nodes: Vec<(Node, i64, usize)>,
    pub(crate) inputs: HashMap<(Node, i64, usize), Type>,
    pub(crate) type_table: Rc<HashMap<syn::Ident, Type>>,
    pub(crate) variables: HashSet<syn::Ident>,
    pub(crate) current_assignment: Option<syn::Ident>,
    pub(crate) current_idx: usize,
    pub(crate) current_block: usize,
    pub(crate) extra_temps: Vec<syn::Ident>,
    pub(crate) errors: Vec<Error>,
}

impl Graph {
    pub fn new(type_table: Rc<HashMap<syn::Ident, Type>>, current_block: usize) -> Self {
        Self {
            type_table,
            nodes: vec![],
            inputs: HashMap::new(),
            variables: HashSet::new(),
            current_assignment: None,
            current_idx: 0,
            current_block,
            extra_temps: vec![],
            errors: vec![],
        }
    }

    pub(crate) fn to_cmp_pet_graph(&self) -> StableGraph<CmpNode, ()> {
        let mut graph: StableGraph<&Operand, ()> = StableGraph::new();
        let mut node_index_map = HashMap::new();
        for node in &self.nodes {
            let data = match node {
                (Node::Unary(unary), _, _) => { &unary.output }
                (Node::Binary(binary), _, _) => { &binary.output }
                (Node::Input(input), _, _) => { input }
            };
            let index = graph.add_node(data);
            node_index_map.insert(data, index);
        }
        for node in &self.nodes {
            match node {
                (Node::Unary(unary), _, _) => {
                    if !node_index_map.contains_key(&unary.operand) && unary.operand.is_variable() {
                        let index = graph.add_node(&unary.operand);
                        node_index_map.insert(&unary.operand, index);
                    }
                }
                (Node::Binary(binary), _, _) => {
                    if !node_index_map.contains_key(&binary.left) && binary.left.is_variable() {
                        let index = graph.add_node(&binary.left);
                        node_index_map.insert(&binary.left, index);
                    }
                    if !node_index_map.contains_key(&binary.right) && binary.right.is_variable() {
                        let index = graph.add_node(&binary.right);
                        node_index_map.insert(&binary.right, index);
                    }
                }
                _ => {}
            }
        }
        drop(graph);
        let mut graph: StableGraph<CmpNode, ()> = StableGraph::new();
        let mut added_nodes = HashSet::new();
        for node in &self.nodes {
            let data = match node {
                (Node::Unary(unary), stmt_idx, block_idx) => {
                    CmpNode {
                        kernel_type: unary.kernel_type,
                        args: vec![node_index_map[&unary.operand]],
                        args_ident: vec![unary.operand.clone()],
                        outputs: vec![],
                        outputs_ident: vec![],
                        stmt_idx: *stmt_idx,
                        block_idx: *block_idx,
                        id: node_index_map[&unary.output],
                        ident: unary.output.clone(),
                        method: Some(unary.method.clone()),
                    }
                }
                (Node::Binary(binary), stmt_idx, block_idx) => {
                    let mut args = vec![];
                    if binary.left.is_variable() {
                        args.push(node_index_map[&binary.left]);
                    }
                    if binary.right.is_variable() {
                        args.push(node_index_map[&binary.right]);
                    }
                    CmpNode {
                        kernel_type: binary.kernel_type,
                        args,
                        args_ident: vec![binary.left.clone(), binary.right.clone()],
                        outputs: vec![],
                        outputs_ident: vec![],
                        stmt_idx: *stmt_idx,
                        block_idx: *block_idx,
                        id: node_index_map[&binary.output],
                        ident: binary.output.clone(),
                        method: Some(binary.method.clone()),
                    }
                }
                (Node::Input(input), stmt_idx, block_idx) => {
                    CmpNode {
                        kernel_type: KernelType::Unary,
                        args: vec![],
                        outputs: vec![],
                        args_ident: vec![],
                        outputs_ident: vec![],
                        stmt_idx: *stmt_idx,
                        block_idx: *block_idx,
                        id: node_index_map[input],
                        ident: input.clone(),
                        method: None,
                    }
                }
            };
            let idx = graph.add_node(data);
            added_nodes.insert(idx);
        }
        for node in &self.nodes {
            match node {
                (Node::Unary(unary), stmt_idx, block_idx) => {
                    if !added_nodes.contains(&node_index_map[&unary.operand]) {
                        let index = graph.add_node(CmpNode {
                            kernel_type: KernelType::Unary,
                            args: vec![],
                            outputs: vec![],
                            args_ident: vec![],
                            outputs_ident: vec![],
                            stmt_idx: *stmt_idx,
                            block_idx: *block_idx,
                            id: node_index_map[&unary.operand],
                            ident: unary.operand.clone(),
                            method: None,
                        });
                        added_nodes.insert(index);
                    }
                }
                (Node::Binary(binary), stmt_idx, block_idx) => {
                    if binary.left.is_variable() {
                        if !added_nodes.contains(&node_index_map[&binary.left]) {
                            let index = graph.add_node(CmpNode {
                                kernel_type: KernelType::Unary,
                                args: vec![],
                                outputs: vec![],
                                args_ident: vec![],
                                outputs_ident: vec![],
                                stmt_idx: *stmt_idx,
                                block_idx: *block_idx,
                                id: node_index_map[&binary.left],
                                ident: binary.left.clone(),
                                method: None,
                            });
                            added_nodes.insert(index);
                        }
                    }
                    if binary.right.is_variable() {
                        if !added_nodes.contains(&node_index_map[&binary.right]) {
                            let index = graph.add_node(CmpNode {
                                kernel_type: KernelType::Unary,
                                args: vec![],
                                outputs: vec![],
                                args_ident: vec![],
                                outputs_ident: vec![],
                                stmt_idx: *stmt_idx,
                                block_idx: *block_idx,
                                id: node_index_map[&binary.right],
                                ident: binary.right.clone(),
                                method: None,
                            });
                            added_nodes.insert(index);
                        }
                    }
                }
                _ => {}
            }
        }
        let node_indices = graph.node_indices().collect::<Vec<_>>();
        for node_idx in node_indices {
            let inputs = graph[node_idx].args.clone();
            for inp in inputs {
                graph[inp].outputs.push(node_idx);
                let ident = graph[node_idx].ident.clone();
                graph[inp].outputs_ident.push(ident);
                graph.add_edge(inp, node_idx, ());
            }
        }
        graph
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

    fn visit_expr_block(&mut self, _: &'ast syn::ExprBlock) {
        unimplemented!("graph::visit_expr_block");
    }

    fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {
        unimplemented!("graph::visit_expr_closure");
    }

    fn visit_expr_const(&mut self, _: &'ast syn::ExprConst) {
        unimplemented!("graph::visit_expr_const");
    }

    fn visit_expr_for_loop(&mut self, _: &'ast syn::ExprForLoop) {
        unimplemented!("graph::visit_expr_for_loop");
    }

    fn visit_expr_group(&mut self, _: &'ast syn::ExprGroup) {
        unimplemented!("graph::visit_expr_group");
    }

    fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {
        unimplemented!("graph::visit_expr_loop");
    }

    fn visit_expr_match(&mut self, _: &'ast syn::ExprMatch) {
        unimplemented!("graph::visit_expr_match");
    }

    fn visit_expr_raw_addr(&mut self, _: &'ast syn::ExprRawAddr) {
        unimplemented!("graph::visit_expr_raw_addr");
    }

    fn visit_expr_unary(&mut self, unary: &'ast syn::ExprUnary) {
        match unary.op {
            syn::UnOp::Not(_) | syn::UnOp::Neg(_) => {
                let left_ty = handle_expr_type(&unary.expr, &self.type_table);
                if left_ty != Type::Tensor {
                    return;
                }
                let current_assignment = if let Some(current_assignment) = &self.current_assignment {
                    Operand::Variable(current_assignment.clone())
                } else {
                    self.errors.push(Error::ExpectedAssignment(unary.span(), "build graph"));
                    return;
                };
                let left = if let Some(ident) = extract_expr_ident(&unary.expr) {
                    ident
                } else {
                    self.errors.push(Error::ExpectedIdentifier(unary.expr.span(), "build graph"));
                    return;
                };

                let method = match unary.op {
                    syn::UnOp::Not(_) => "not",
                    syn::UnOp::Neg(_) => "neg",
                    _ => todo!(),
                };
                self.nodes.push((
                    Node::Unary(Unary {
                        method: proc_macro2::Ident::new(method, unary.span()),
                        operand: Operand::Variable(left.clone()),
                        args: vec![],
                        output: current_assignment.clone(),
                        kernel_type: KernelType::Unary,
                    }),
                    self.current_idx as i64,
                    self.current_block,
                ));
            }
            _ => {}
        }
    }

    fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {
        unimplemented!("graph::visit_expr_while");
    }

    fn visit_field(&mut self, _: &'ast syn::Field) {
        unimplemented!("graph::visit_field");
    }

    fn visit_field_mutability(&mut self, _: &'ast syn::FieldMutability) {
        unimplemented!("graph::visit_field_mutability");
    }

    fn visit_fields(&mut self, _: &'ast syn::Fields) {}

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

    fn visit_item_macro(&mut self, _: &'ast syn::ItemMacro) {}

    fn visit_item_mod(&mut self, _: &'ast syn::ItemMod) {
        unimplemented!("graph::visit_item_mod");
    }

    fn visit_item_static(&mut self, _: &'ast syn::ItemStatic) {
        unimplemented!("graph::visit_item_static");
    }

    fn visit_item_struct(&mut self, _: &'ast syn::ItemStruct) {}

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
        match &local.pat {
            syn::Pat::Ident(pat_ident) => {
                self.visit_pat_ident(pat_ident);
                self.current_assignment = Some(pat_ident.ident.clone());
                if let Some(it) = &local.init {
                    if let syn::Expr::Reference(expr_reference) = it.expr.as_ref() {
                        let ty = handle_expr_type(&expr_reference.expr, &self.type_table);
                        if ty == Type::Tensor {
                            self.inputs.insert(
                                (
                                    Node::Input(Operand::Variable(pat_ident.ident.clone())),
                                    self.current_idx as i64,
                                    self.current_block,
                                ),
                                ty
                            );
                            self.variables.insert(pat_ident.ident.clone());
                        }
                    } else {
                        self.visit_local_init(it);
                        self.variables.insert(pat_ident.ident.clone());
                    }
                }
            }
            _ => {}
        }
        self.current_assignment = None;
    }
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let receiver_ty = handle_expr_type(&node.receiver, &self.type_table);
        if receiver_ty != Type::Tensor {
            return;
        }
        let method_name = node.method.to_string();
        let is_binary = BINARY_OPERATORS.contains(&method_name.as_str());
        let is_opaque_binary = OPAQUE_BINARY_OPERATORS.contains(&method_name.as_str());
        let is_unary = UNARY_OPERATORS.contains(&method_name.as_str());
        if !is_unary && !is_binary && !is_opaque_binary {
            return;
        }
        let current_assignment = if let Some(assingment) = self.current_assignment.clone() {
            assingment
        } else {
            return;
        };
        let receiver_var = if let Some(ident) = extract_expr_ident(&node.receiver) {
            Operand::Variable(ident)
        } else {
            self.errors.push(Error::ExpectedPath(node.receiver.span(), "build graph"));
            return;
        };
        let mut args = vec![];
        for arg in &node.args {
            if let Some(ident) = extract_expr_ident(arg) {
                args.push(Operand::Variable(ident));
            } else {
                self.errors.push(Error::ExpectedIdentifier(arg.span(), "build graph"));
                return;
            }
        }
        if is_unary {
            let method = Node::Unary(Unary {
                method: node.method.clone(),
                operand: receiver_var.clone(),
                args,
                output: Operand::Variable(current_assignment.clone()),
                kernel_type: KernelType::Unary,
            });
            self.nodes.push((method, self.current_idx as i64, self.current_block));
        } else {
            if is_binary || is_opaque_binary {
                let right = if let Some(ident) = extract_expr_ident(&node.args[0]) {
                    Operand::Variable(ident)
                } else {
                    self.errors.push(Error::ExpectedIdentifier(node.args[0].span(), "build graph"));
                    return;
                };
                let method = Node::Binary(Binary {
                    method: node.method.clone(),
                    left: receiver_var.clone(),
                    right: right.clone(),
                    output: Operand::Variable(current_assignment.clone()),
                    kernel_type: if is_binary {
                        KernelType::Binary
                    } else {
                        KernelType::Opaque
                    },
                });
                self.nodes.push((method, self.current_idx as i64, self.current_block));
            }
        }
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
        let current_assignment = if let Some(current_assignment) = &self.current_assignment {
            current_assignment.clone()
        } else {
            self.errors.push(Error::ExpectedAssignment(node.span(), "build graph"));
            return;
        };
        let left = if let Some(ident) = extract_expr_ident(&node.left) {
            Operand::Variable(ident)
        } else if let syn::Expr::Lit(lit) = node.left.as_ref() {
            Operand::Constant(syn::Expr::Lit(lit.clone()))
        } else {
            self.errors.push(Error::ExpectedIdentifier(node.left.span(), "build graph"));
            return;
        };
        let right = if let Some(ident) = extract_expr_ident(&node.right) {
            Operand::Variable(ident)
        } else if let syn::Expr::Lit(lit) = node.right.as_ref() {
            Operand::Constant(syn::Expr::Lit(lit.clone()))
        } else {
            self.errors.push(Error::ExpectedIdentifier(node.right.span(), "build graph"));
            return;
        };

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
            _ => unimplemented!("build_graph::visit_expr_binary"),
        };

        self.nodes.push((
            Node::Binary(Binary {
                method: proc_macro2::Ident::new(method, node.span()),
                left: left.clone(),
                right: right.clone(),
                output: Operand::Variable(current_assignment.clone()),
                kernel_type: KernelType::Binary,
            }),
            self.current_idx as i64,
            self.current_block,
        ));
    }
}

fn handle_expr_type<'ast>(node: &'ast syn::Expr, type_table: &HashMap<syn::Ident, Type>) -> Type {
    match node {
        syn::Expr::Path(expr_path) => {
            if let Some(ident) = expr_path.path.get_ident() {
                type_table.get(ident).unwrap_or(&Type::Unknown).clone()
            } else {
                Type::Unknown
            }
        }
        syn::Expr::Reference(expr_reference) => {
            handle_expr_type(&expr_reference.expr, type_table)
        }
        _ => Type::Unknown,
    }
}

fn extract_expr_ident(node: &syn::Expr) -> Option<syn::Ident> {
    match node {
        syn::Expr::Path(expr_path) => expr_path.path.get_ident().cloned(),
        syn::Expr::Reference(expr_reference) => extract_expr_ident(&expr_reference.expr),
        _ => None,
    }
}
