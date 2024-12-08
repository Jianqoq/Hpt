use std::collections::{ HashMap, HashSet };

use petgraph::graph::NodeIndex;
use quote::ToTokens;
use syn::{ spanned::Spanned, visit::Visit };

use super::{ cfg::{ BasicBlock, BlockId, BlockType, CustomStmt, CFG }, errors::Error };

pub(crate) struct CFGBuilder<'a> {
    pub(crate) cfg: &'a mut CFG,
    pub(crate) current_block: NodeIndex,
    pub(crate) block_ids: BlockId,
    pub(crate) current_expr: Option<syn::Expr>,
    pub(crate) current_pat: Option<syn::Pat>,
    pub(crate) current_type: Option<syn::Type>,
    pub(crate) current_item: Option<syn::Item>,
    pub(crate) is_last_stmt: bool,
    pub(crate) global_block_cnt: usize,
    pub(crate) errors: Vec<Error>,
    pub(crate) has_assignment: bool,
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
            is_last_stmt: false,
            current_pat: None,
            current_type: None,
            current_item: None,
            errors: vec![],
            has_assignment: false,
        }
    }

    fn global_block_cnt(&mut self) -> usize {
        let cnt = self.global_block_cnt;
        self.global_block_cnt += 1;
        cnt
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

    /// connect current block to target block
    fn connect_to(&mut self, to: NodeIndex) {
        self.cfg.connect(self.current_block, to);
    }

    fn set_current_block(&mut self, new_block: NodeIndex) {
        self.current_block = new_block;
    }

    fn set_current_block_id(&mut self, new_block_id: BlockId) {
        self.block_ids = new_block_id;
    }

    /// handle if statement
    fn handle_if(&mut self, expr_if: &syn::ExprIf) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let assign_block = self.new_block(BlockType::IfAssign);
        let assign_block_id = BlockId::new(assign_block);
        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let assign_ident = syn::Ident::new(
            &format!("__if_assign_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        self.cfg.graph[assign_block].statements.push(CustomStmt {
            stmt: syn::parse2(quote::quote! { let #assign_ident; }).expect("assign stmt is none"),
        });
        let cond_block = self.new_block(BlockType::IfCond);
        let cond_block_id = BlockId::new(cond_block);

        // create then branch block
        let then_block = self.new_block(BlockType::IfThenEnd);
        let then_block_id = BlockId::new(then_block);
        // create else branch block
        let else_block = self.new_block(BlockType::IfElseEnd);
        let else_block_id = BlockId::new(else_block);
        // create merge block
        let merge_block = self.new_block(BlockType::Normal);
        let merge_block_id = BlockId::new(merge_block);

        // connect current block to condition check block
        self.connect_to(cond_block);
        self.set_current_block(cond_block);
        self.set_current_block_id(cond_block_id);
        self.push_stmt(syn::Stmt::Expr(*expr_if.cond.clone(), None));
        let cond_block_id = core::mem::take(&mut self.block_ids);
        // connect current block to then and else branch
        self.connect_to(then_block);
        self.connect_to(else_block);

        self.set_current_block_id(then_block_id);
        // handle then branch
        self.set_current_block(then_block);
        self.visit_block(&expr_if.then_branch);
        let then_block_id = core::mem::take(&mut self.block_ids);
        self.connect_to(merge_block);
        // handle else branch
        self.set_current_block_id(else_block_id);
        self.set_current_block(else_block);
        if let Some(else_branch) = &expr_if.else_branch {
            self.cfg.graph[then_block].block_type = BlockType::IfThen;
            match &else_branch.1.as_ref() {
                syn::Expr::Block(expr_block) => {
                    self.visit_block(&expr_block.block);
                }
                syn::Expr::If(expr_if) => {
                    self.handle_if(expr_if);
                    if let Some(expr) = self.current_expr.take() {
                        self.push_stmt(syn::Stmt::Expr(expr, None));
                    }
                }
                _ => {
                    self.visit_expr(&else_branch.1);
                }
            }
        }
        let else_block_id = core::mem::take(&mut self.block_ids);
        self.connect_to(merge_block);

        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(cond_block_id);
        current_block_id.children.push(then_block_id);
        current_block_id.children.push(else_block_id);
        current_block_id.children.push(merge_block_id);
        self.set_current_block(merge_block);
        self.set_current_block_id(current_block_id);

        if let Ok(expr) = syn::parse2(quote::quote! { #assign_ident }) {
            self.current_expr = Some(expr);
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_if.span(),
                    "CFG builder",
                    format!("{}", assign_ident.to_string())
                )
            );
        }
    }

    /// handle loop statement
    fn handle_loop(&mut self, expr_loop: &syn::ExprLoop) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let assign_block = self.new_block(BlockType::LoopAssign);
        let assign_block_id = BlockId::new(assign_block);
        // create loop body block
        let loop_block = self.new_block(BlockType::LoopBody);
        let loop_block_id = BlockId::new(loop_block);
        // create after loop block
        let after_loop_block = self.new_block(BlockType::Normal);
        let after_loop_block_id = BlockId::new(after_loop_block);

        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let assign_ident = syn::Ident::new(
            &format!("__loop_assign_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        if let Ok(stmt) = syn::parse2(quote::quote! { let #assign_ident; }) {
            self.cfg.graph[assign_block].statements.push(CustomStmt {
                stmt,
            });
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_loop.span(),
                    "CFG builder",
                    format!("let {};", assign_ident.to_string())
                )
            );
            return;
        }

        // connect current block to loop body block
        self.connect_to(loop_block);

        // set current block to loop body block and handle loop body
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_loop.body);
        if let Ok(expr) = syn::parse2(quote::quote! { #assign_ident }) {
            self.current_expr = Some(expr);
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_loop.span(),
                    "CFG builder",
                    format!("{}", assign_ident.to_string())
                )
            );
        }
        let loop_block_id = core::mem::take(&mut self.block_ids);

        // connect loop body block to itself, represent next iteration
        self.connect_to(loop_block);
        // connect loop body block to after loop block, represent exit loop
        self.connect_to(after_loop_block);

        // set current block to after loop block and handle next statements
        self.set_current_block(after_loop_block);
        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(loop_block_id);
        current_block_id.children.push(after_loop_block_id);
        self.set_current_block_id(current_block_id);
    }

    // handle match statement
    fn handle_match(&mut self, expr_match: &syn::ExprMatch) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let cond_block = self.new_block(BlockType::MatchCond);
        let cond_block_id = BlockId::new(cond_block);
        // create merge block
        let merge_block = self.new_block(BlockType::Normal);
        let merge_block_id = BlockId::new(merge_block);

        self.connect_to(cond_block);
        self.set_current_block(cond_block);
        self.push_stmt(syn::Stmt::Expr(*expr_match.expr.clone(), None));

        for arm in &expr_match.arms {
            // 为每个匹配分支创建一个块
            let case_block = self.new_block(BlockType::MatchCase);
            let case_block_id = BlockId::new(case_block);
            // 连接当前块到分支块
            self.connect_to(case_block);
            // 处理分支块中的语句
            self.set_current_block(case_block);
            let pat = &arm.pat;
            if let Ok(stmt) = syn::parse2(quote::quote! { let __pat = #pat; }) {
                self.push_stmt(stmt);
            } else {
                self.errors.push(
                    Error::SynParseError(
                        arm.pat.span(),
                        "CFG builder",
                        format!("{}", arm.pat.to_token_stream().to_string())
                    )
                );
            }
            let body_block = self.new_block(BlockType::Normal);
            let body_block_id = BlockId::new(body_block);
            self.connect_to(body_block);
            self.set_current_block(body_block);
            self.set_current_block_id(body_block_id);
            self.visit_expr(&arm.body);
            let body_block_id = core::mem::take(&mut self.block_ids);
            current_block_id.children.push(case_block_id);
            current_block_id.children.push(body_block_id);
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
            &format!("__block_out_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        if let Ok(stmt) = syn::parse2(quote::quote! { let #block_ident; }) {
            self.cfg.graph[assign_block].statements.push(CustomStmt {
                stmt,
            });
        } else {
            self.errors.push(
                Error::SynParseError(
                    block.span(),
                    "CFG builder",
                    format!("let {};", block_ident.to_string())
                )
            );
            return;
        }
        self.connect_to(new_block);
        self.set_current_block(new_block);
        self.set_current_block_id(new_block_id);
        self.visit_block(&block.block);
        if let Ok(expr) = syn::parse2(quote::quote! { #block_ident }) {
            self.current_expr = Some(expr);
        } else {
            self.errors.push(
                Error::SynParseError(
                    block.span(),
                    "CFG builder",
                    format!("{}", block_ident.to_string())
                )
            );
        }
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
        let assign_block = self.new_block(BlockType::WhileAssign);
        let assign_block_id = BlockId::new(assign_block);
        // 创建条件检查块
        let condition_block = self.new_block(BlockType::WhileCond);
        let condition_block_id = BlockId::new(condition_block);
        // 创建循环体块
        let loop_block = self.new_block(BlockType::WhileBody);
        let loop_block_id = BlockId::new(loop_block);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);
        let after_loop_block_id = BlockId::new(after_loop_block);

        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let block_ident = syn::Ident::new(
            &format!("__while_out_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        if let Ok(stmt) = syn::parse2(quote::quote! { let #block_ident; }) {
            self.cfg.graph[assign_block].statements.push(CustomStmt {
                stmt,
            });
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_while.span(),
                    "CFG builder",
                    format!("let {};", block_ident.to_string())
                )
            );
            return;
        }

        // 连接当前块到条件检查块
        self.connect_to(condition_block);

        // 连接条件检查块到循环体块（如果条件为真）和循环后的块（如果条件为假）
        self.set_current_block(condition_block);
        self.push_stmt(syn::Stmt::Expr(*expr_while.cond.clone(), None));

        // 创建两个新的连接：条件为真和条件为假
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_while.body);
        if let Ok(expr) = syn::parse2(quote::quote! { #block_ident }) {
            self.current_expr = Some(expr);
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_while.span(),
                    "CFG builder",
                    format!("{}", block_ident.to_string())
                )
            );
        }
        let loop_block_id = core::mem::take(&mut self.block_ids);

        // 连接循环体块回到条件检查块（表示下一次迭代）
        self.connect_to(condition_block);
        // 连接循环体块到循环后的块（如果有 break）
        self.connect_to(after_loop_block);

        // 设置当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(condition_block_id);
        current_block_id.children.push(loop_block_id);
        current_block_id.children.push(after_loop_block_id);
        self.set_current_block_id(current_block_id);
    }

    fn handle_for_loop(&mut self, expr_for: &syn::ExprForLoop) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let assign_block = self.new_block(BlockType::ForAssign);
        let assign_block_id = BlockId::new(assign_block);
        // create iterator init block
        let init_block = self.new_block(BlockType::ForInit);
        let init_block_id = BlockId::new(init_block);
        // create condition check block
        let condition_block = self.new_block(BlockType::ForCond);
        let condition_block_id = BlockId::new(condition_block);
        // create loop body block
        let loop_block = self.new_block(BlockType::ForBody);
        let loop_block_id = BlockId::new(loop_block);
        // create after loop block
        let after_loop_block = self.new_block(BlockType::Normal);
        let after_loop_block_id = BlockId::new(after_loop_block);

        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let block_ident = syn::Ident::new(
            &format!("__for_out_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        if let Ok(stmt) = syn::parse2(quote::quote! { let #block_ident; }) {
            self.cfg.graph[assign_block].statements.push(CustomStmt {
                stmt,
            });
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_for.span(),
                    "CFG builder",
                    format!("let {};", block_ident.to_string())
                )
            );
            return;
        }

        // connect current block to init block
        self.connect_to(init_block);

        // handle iterator init and element binding
        self.set_current_block(init_block);
        self.set_current_block_id(init_block_id);
        let init_block_id = core::mem::take(&mut self.block_ids);
        let pat = &expr_for.pat;
        let local = quote::quote! {
            let #pat;
        };
        if let Ok(stmt) = syn::parse2(local) {
            self.push_stmt(stmt);
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_for.span(),
                    "CFG builder",
                    format!("let {};", pat.to_token_stream().to_string())
                )
            );
            return;
        }
        // connect init block to condition check block
        self.connect_to(condition_block);

        // handle condition check
        self.set_current_block(condition_block);
        self.push_stmt(syn::Stmt::Expr(*expr_for.expr.clone(), None));
        // create connection: condition is true enter loop body block, condition is false enter after loop block
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // handle loop body
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_for.body);
        if let Ok(expr) = syn::parse2(quote::quote! { #block_ident }) {
            self.current_expr = Some(expr);
        } else {
            self.errors.push(
                Error::SynParseError(
                    expr_for.span(),
                    "CFG builder",
                    format!("{}", block_ident.to_string())
                )
            );
            return;
        }
        let loop_block_id = core::mem::take(&mut self.block_ids);
        // connect loop body block to condition check block (represent next iteration)
        self.connect_to(condition_block);
        // connect loop body block to after loop block (if there is break)
        self.connect_to(after_loop_block);

        // set current block to after loop block and handle next statements
        self.set_current_block(after_loop_block);
        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(init_block_id);
        current_block_id.children.push(condition_block_id);
        current_block_id.children.push(loop_block_id);
        current_block_id.children.push(after_loop_block_id);
        self.set_current_block_id(current_block_id);
    }

    fn push_stmt(&mut self, stmt: syn::Stmt) {
        self.cfg.graph[self.current_block].statements.push(CustomStmt { stmt });
    }
}

impl<'ast, 'a> syn::visit::Visit<'ast> for CFGBuilder<'a> {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let visibility_block = self.new_block(BlockType::FnVisibility(i.vis.clone()));
        let visibility_block_id = BlockId::new(visibility_block);
        let args_block = self.new_block(BlockType::FnArgs);
        let args_block_id = BlockId::new(args_block);
        let name_block = self.new_block(BlockType::FnName);
        let name_block_id = BlockId::new(name_block);
        let generics_block = self.new_block(BlockType::Generics(i.sig.generics.clone()));
        let generics_block_id = BlockId::new(generics_block);
        let ret_block = self.new_block(BlockType::FnRet(i.sig.output.clone()));
        let ret_block_id = BlockId::new(ret_block);
        let body_block = self.new_block(BlockType::FnBody);
        let body_block_id = BlockId::new(body_block);
        let (where_block, where_block_id) = if
            let Some(where_clause) = &i.sig.generics.where_clause
        {
            let where_block = self.new_block(BlockType::Where(where_clause.clone()));
            let where_block_id = BlockId::new(where_block);
            (Some(where_block), Some(where_block_id))
        } else {
            (None, None)
        };
        let after_fn_block = self.new_block(BlockType::Normal);
        let after_fn_block_id = BlockId::new(after_fn_block);
        self.connect_to(visibility_block);
        self.set_current_block(visibility_block);
        self.connect_to(name_block);
        self.set_current_block(name_block);
        let name = &i.sig.ident;
        if let Ok(stmt) = syn::parse2(quote::quote! { let #name; }) {
            self.push_stmt(stmt);
        } else {
            self.errors.push(
                Error::SynParseError(
                    i.sig.ident.span(),
                    "CFG builder",
                    format!("let {};", name.to_string())
                )
            );
        }
        self.connect_to(args_block);
        self.set_current_block(args_block);
        for arg in &i.sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => {
                    self.errors.push(
                        Error::Unsupported(
                            arg.span(),
                            "CFG builder",
                            "function receiver".to_string()
                        )
                    );
                    return;
                }
                syn::FnArg::Typed(pat_type) => {
                    let arg_stmt = syn
                        ::parse2(quote::quote! { let #pat_type; })
                        .expect("arg stmt is none::365");
                    self.push_stmt(arg_stmt);
                }
            }
        }
        self.connect_to(ret_block);
        self.set_current_block(ret_block);
        if let Some(where_block) = where_block {
            self.connect_to(where_block);
            self.set_current_block(where_block);
        }
        self.connect_to(body_block);
        self.set_current_block(body_block);
        self.set_current_block_id(body_block_id);
        self.visit_block(&i.block);
        let body_block_id = core::mem::take(&mut self.block_ids);
        current_block_id.children.push(visibility_block_id);
        current_block_id.children.push(name_block_id);
        current_block_id.children.push(generics_block_id);
        current_block_id.children.push(args_block_id);
        current_block_id.children.push(ret_block_id);
        if let Some(where_block_id) = where_block_id {
            current_block_id.children.push(where_block_id);
        }
        current_block_id.children.push(body_block_id);
        current_block_id.children.push(after_fn_block_id);
        self.set_current_block_id(current_block_id);
        self.connect_to(after_fn_block);
        self.set_current_block(after_fn_block);
    }

    fn visit_fn_arg(&mut self, arg: &'ast syn::FnArg) {
        match arg {
            syn::FnArg::Receiver(_) => {
                self.errors.push(
                    Error::Unsupported(arg.span(), "CFG builder", "function receiver".to_string())
                );
                return;
            }
            syn::FnArg::Typed(pat_type) => {
                self.visit_pat_type(pat_type);
            }
        }
    }

    fn visit_pat_ident(&mut self, i: &'ast syn::PatIdent) {
        self.current_pat = Some(syn::Pat::Ident(i.clone()));
    }

    fn visit_type(&mut self, i: &'ast syn::Type) {
        self.current_type = Some(i.clone());
    }

    fn visit_pat_type(&mut self, i: &'ast syn::PatType) {
        let mut new_pat = i.clone();
        self.visit_pat(i.pat.as_ref());
        if let Some(pat) = core::mem::take(&mut self.current_pat) {
            new_pat.pat = Box::new(pat);
        } else {
            self.errors.push(
                Error::ExprAccumulateError(i.pat.span(), "CFG builder", "pat type".to_string())
            );
            return;
        }
        self.visit_type(&i.ty);
        handle_accumulation(&mut self.errors, i.ty.span(), "pat type", &mut self.current_type).map(
            |ty| {
                new_pat.ty = Box::new(ty);
                self.current_pat = Some(syn::Pat::Type(new_pat));
            }
        );
    }

    fn visit_expr_range(&mut self, i: &'ast syn::ExprRange) {
        let mut new_expr = i.clone();
        match (new_expr.start.as_mut(), new_expr.end.as_mut()) {
            (None, None) => {
                self.current_expr = Some(syn::Expr::Range(new_expr));
            }
            (None, Some(right)) => {
                self.visit_expr(right);
                handle_accumulation(
                    &mut self.errors,
                    i.end.span(),
                    "range",
                    &mut self.current_expr
                ).map(|right| {
                    new_expr.end = Some(Box::new(right));
                    self.current_expr = Some(syn::Expr::Range(new_expr));
                });
            }
            (Some(left), None) => {
                self.visit_expr(left);
                handle_accumulation(
                    &mut self.errors,
                    i.start.span(),
                    "range",
                    &mut self.current_expr
                ).map(|start| {
                    new_expr.start = Some(Box::new(start));
                    self.current_expr = Some(syn::Expr::Range(new_expr));
                });
            }
            (Some(left), Some(right)) => {
                self.visit_expr(left);
                let start_status = handle_accumulation(
                    &mut self.errors,
                    i.start.span(),
                    "range",
                    &mut self.current_expr
                ).map(|start| {
                    new_expr.start = Some(Box::new(start));
                });
                self.visit_expr(right);
                let end_status = handle_accumulation(
                    &mut self.errors,
                    i.end.span(),
                    "range",
                    &mut self.current_expr
                ).map(|right| {
                    new_expr.end = Some(Box::new(right));
                });
                if start_status.is_some() && end_status.is_some() {
                    self.current_expr = Some(syn::Expr::Range(new_expr));
                }
            }
        }
    }

    fn visit_expr_break(&mut self, i: &'ast syn::ExprBreak) {
        self.current_expr = Some(syn::Expr::Break(i.clone()));
    }

    fn visit_expr_continue(&mut self, i: &'ast syn::ExprContinue) {
        self.current_expr = Some(syn::Expr::Continue(i.clone()));
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

    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        self.current_expr = Some(syn::Expr::Path(i.clone()));
    }

    fn visit_expr_reference(&mut self, i: &'ast syn::ExprReference) {
        let mut new_expr = i.clone();
        self.visit_expr(i.expr.as_ref());

        handle_accumulation(
            &mut self.errors,
            i.expr.span(),
            "reference",
            &mut self.current_expr
        ).map(|expr| {
            new_expr.expr = Box::new(expr);
            let new_expr = syn::Expr::Reference(new_expr);
            self.current_expr = Some(new_expr);
        });
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        self.has_assignment = false;
        let mut new_binary = binary.clone();

        match (new_binary.left.as_mut(), new_binary.right.as_mut()) {
            (syn::Expr::Binary(left), syn::Expr::Binary(right)) => {
                self.visit_expr_binary(left);
                let left = handle_accumulation(
                    &mut self.errors,
                    left.span(),
                    "binary",
                    &mut self.current_expr
                );
                let left_stmt = if let Some(left) = left {
                    if
                        let Ok(stmt) = syn::parse2::<syn::Stmt>(
                            quote::quote! { let __out_1 = #left; }
                        )
                    {
                        stmt
                    } else {
                        self.errors.push(
                            Error::SynParseError(left.span(), "CFG builder", "binary".to_string())
                        );
                        return;
                    }
                } else {
                    return;
                };
                self.visit_expr_binary(right);
                let right = handle_accumulation(
                    &mut self.errors,
                    right.span(),
                    "binary",
                    &mut self.current_expr
                );
                let right_stmt = if let Some(right) = right {
                    if
                        let Ok(stmt) = syn::parse2::<syn::Stmt>(
                            quote::quote!(let __out_2 = #right;)
                        )
                    {
                        stmt
                    } else {
                        self.errors.push(
                            Error::SynParseError(right.span(), "CFG builder", "binary".to_string())
                        );
                        return;
                    }
                } else {
                    return;
                };
                let op = &binary.op;
                if let Ok(expr) = syn::parse2::<syn::Expr>(quote::quote!(__out_1 #op __out_2)) {
                    self.current_expr = Some(expr);
                } else {
                    self.errors.push(
                        Error::SynParseError(
                            binary.span(),
                            "CFG builder",
                            format!("__out_1 {} __out_2", op.to_token_stream().to_string())
                        )
                    );
                    return;
                }
                self.push_stmt(left_stmt);
                self.push_stmt(right_stmt);
            }
            (syn::Expr::Binary(left), _) => {
                self.visit_expr_binary(left);
                let left = handle_accumulation(
                    &mut self.errors,
                    left.span(),
                    "binary",
                    &mut self.current_expr
                );
                let left_stmt = if let Some(left) = left {
                    if
                        let Ok(stmt) = syn::parse2::<syn::Stmt>(
                            quote::quote! { let __out_1 = #left; }
                        )
                    {
                        stmt
                    } else {
                        self.errors.push(
                            Error::SynParseError(left.span(), "CFG builder", "binary".to_string())
                        );
                        return;
                    }
                } else {
                    return;
                };
                self.visit_expr(binary.right.as_ref());
                let right = handle_accumulation(
                    &mut self.errors,
                    binary.right.span(),
                    "binary",
                    &mut self.current_expr
                );
                let right = if let Some(right) = right {
                    right
                } else {
                    return;
                };
                let op = &binary.op;
                if let Ok(expr) = syn::parse2::<syn::Expr>(quote::quote!(__out_1 #op #right)) {
                    self.current_expr = Some(expr);
                } else {
                    self.errors.push(
                        Error::SynParseError(
                            binary.span(),
                            "CFG builder",
                            format!(
                                "__out_1 {} {}",
                                op.to_token_stream().to_string(),
                                right.to_token_stream().to_string()
                            )
                        )
                    );
                    return;
                }
                self.push_stmt(left_stmt);
            }
            (_, syn::Expr::Binary(right)) => {
                self.visit_expr(binary.left.as_ref());
                let left = handle_accumulation(
                    &mut self.errors,
                    binary.left.span(),
                    "binary",
                    &mut self.current_expr
                );
                let left = if let Some(left) = left {
                    left
                } else {
                    return;
                };
                self.visit_expr_binary(right);
                let right = handle_accumulation(
                    &mut self.errors,
                    binary.right.span(),
                    "binary",
                    &mut self.current_expr
                );
                let right_stmt = if let Some(right) = right {
                    if
                        let Ok(stmt) = syn::parse2::<syn::Stmt>(
                            quote::quote!(let __out_2 = #right;)
                        )
                    {
                        stmt
                    } else {
                        self.errors.push(
                            Error::SynParseError(right.span(), "CFG builder", "binary".to_string())
                        );
                        return;
                    }
                } else {
                    return;
                };
                let op = &binary.op;
                if let Ok(expr) = syn::parse2::<syn::Expr>(quote::quote!(#left #op __out_2)) {
                    self.current_expr = Some(expr);
                } else {
                    self.errors.push(
                        Error::SynParseError(
                            binary.span(),
                            "CFG builder",
                            format!(
                                "{} {} __out_2",
                                left.to_token_stream().to_string(),
                                op.to_token_stream().to_string()
                            )
                        )
                    );
                    return;
                }
                self.push_stmt(right_stmt);
            }
            _ => {
                self.visit_expr(binary.left.as_ref());
                let left = handle_accumulation(
                    &mut self.errors,
                    binary.left.span(),
                    "binary",
                    &mut self.current_expr
                );
                let left = if let Some(left) = left {
                    left
                } else {
                    return;
                };
                self.visit_expr(binary.right.as_ref());
                let right = handle_accumulation(
                    &mut self.errors,
                    binary.right.span(),
                    "binary",
                    &mut self.current_expr
                );
                let right = if let Some(right) = right {
                    right
                } else {
                    return;
                };
                let op = &binary.op;
                if let Ok(expr) = syn::parse2::<syn::Expr>(quote::quote!(#left #op #right)) {
                    self.current_expr = Some(expr);
                } else {
                    self.errors.push(
                        Error::SynParseError(
                            binary.span(),
                            "CFG builder",
                            format!(
                                "{} {} {}",
                                left.to_token_stream().to_string(),
                                op.to_token_stream().to_string(),
                                right.to_token_stream().to_string()
                            )
                        )
                    );
                }
            }
        }
    }

    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        self.current_expr = Some(syn::Expr::Lit(i.clone()));
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let has_assignment = self.has_assignment;
        let mut new_call = call.clone();
        self.has_assignment = false;
        self.visit_expr(call.func.as_ref());
        let func = core::mem::take(&mut self.current_expr).expect("func is none");
        for arg in new_call.args.iter_mut() {
            self.visit_expr(arg);
            let new_arg = handle_accumulation(
                &mut self.errors,
                arg.span(),
                "expr_call",
                &mut self.current_expr
            );
            let new_arg = if let Some(new_arg) = new_arg {
                new_arg
            } else {
                return;
            };
            *arg = new_arg;
        }
        new_call.func = Box::new(func);
        if has_assignment {
            self.current_expr = Some(syn::Expr::Call(new_call));
        } else {
            let ident = syn::Ident::new(
                &format!("__call_{}", self.global_block_cnt()),
                proc_macro2::Span::call_site()
            );
            if let Ok(stmt) = syn::parse2(quote::quote! { let #ident = #new_call; }) {
                self.push_stmt(stmt);
            }
            self.current_expr = Some(syn::parse2(quote::quote! { #ident }).expect("expr is none"));
        }
    }
    fn visit_expr_method_call(&mut self, method_call: &'ast syn::ExprMethodCall) {
        let has_assignment = self.has_assignment;
        let mut new_method_call = method_call.clone();
        self.has_assignment = false;
        self.visit_expr(method_call.receiver.as_ref());
        let receiver = handle_accumulation(
            &mut self.errors,
            method_call.receiver.span(),
            "expr_method_call",
            &mut self.current_expr
        );
        let receiver = if let Some(receiver) = receiver {
            receiver
        } else {
            return;
        };
        match receiver {
            syn::Expr::Binary(_) | syn::Expr::Reference(_) | syn::Expr::Tuple(_) => {
                if let Ok(stmt) = syn::parse2(quote::quote!(let __expr_receiver = #receiver;)) {
                    self.push_stmt(stmt);
                } else {
                    self.errors.push(
                        Error::SynParseError(
                            receiver.span(),
                            "CFG builder",
                            "expr_method_call::stmt".to_string()
                        )
                    );
                    return;
                }
                if let Ok(expr) = syn::parse2(quote::quote! { __expr_receiver }) {
                    self.current_expr = Some(expr);
                } else {
                    self.errors.push(
                        Error::SynParseError(
                            receiver.span(),
                            "CFG builder",
                            "expr_method_call::expr".to_string()
                        )
                    );
                    return;
                }
            }
            _ => {
                new_method_call.receiver = Box::new(receiver);
            }
        }
        for arg in new_method_call.args.iter_mut() {
            self.visit_expr(arg);
            let new_arg = handle_accumulation(
                &mut self.errors,
                arg.span(),
                "expr_method_call",
                &mut self.current_expr
            );
            let new_arg = if let Some(new_arg) = new_arg {
                new_arg
            } else {
                return;
            };
            *arg = new_arg;
        }
        if has_assignment {
            self.current_expr = Some(syn::Expr::MethodCall(new_method_call));
        } else {
            let ident = syn::Ident::new(
                &format!("__method_call_{}", self.global_block_cnt()),
                proc_macro2::Span::call_site()
            );
            if let Ok(stmt) = syn::parse2(quote::quote! { let #ident = #new_method_call; }) {
                self.push_stmt(stmt);
            }
            self.current_expr = Some(syn::parse2(quote::quote! { #ident }).expect("expr is none"));
        }
    }

    fn visit_expr_tuple(&mut self, tuple: &'ast syn::ExprTuple) {
        let mut new_tuple = tuple.clone();
        for elem in new_tuple.elems.iter_mut() {
            self.visit_expr(elem);
            let new_elem = handle_accumulation(
                &mut self.errors,
                elem.span(),
                "expr_tuple",
                &mut self.current_expr
            );
            let new_elem = if let Some(new_elem) = new_elem {
                new_elem
            } else {
                return;
            };
            *elem = new_elem;
        }
        let ident = syn::Ident::new(
            &format!("__expr_tuple_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        if let Ok(stmt) = syn::parse2(quote::quote! { let #ident = #new_tuple; }) {
            self.push_stmt(stmt);
        } else {
            self.errors.push(
                Error::SynParseError(
                    new_tuple.span(),
                    "CFG builder",
                    "expr_tuple::stmt".to_string()
                )
            );
            return;
        }
        if let Ok(expr) = syn::parse2(quote::quote! { #ident }) {
            self.current_expr = Some(expr);
        } else {
            self.errors.push(
                Error::SynParseError(
                    new_tuple.span(),
                    "CFG builder",
                    "expr_tuple::expr".to_string()
                )
            );
        }
    }

    fn visit_expr_try(&mut self, i: &'ast syn::ExprTry) {
        let has_assignment = self.has_assignment;
        let mut new_try = i.clone();
        self.has_assignment = false;
        self.visit_expr(&i.expr);
        let expr = handle_accumulation(
            &mut self.errors,
            i.expr.span(),
            "expr_try",
            &mut self.current_expr
        );
        let expr = if let Some(expr) = expr {
            expr
        } else {
            return;
        };
        new_try.expr = Box::new(expr);
        if has_assignment {
            self.current_expr = Some(syn::Expr::Try(new_try));
        } else {
            let ident = syn::Ident::new(
                &format!("__try_{}", self.global_block_cnt()),
                proc_macro2::Span::call_site()
            );
            if let Ok(stmt) = syn::parse2(quote::quote! { let #ident = #new_try; }) {
                self.push_stmt(stmt);
                self.current_expr = Some(
                    syn::parse2(quote::quote! { #ident }).expect("expr is none")
                );
            } else {
                self.errors.push(
                    Error::SynParseError(
                        new_try.span(),
                        "CFG builder",
                        "expr_try::stmt".to_string()
                    )
                );
            }
        }
    }

    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        let mut new_assign = i.clone();
        self.visit_expr(&i.right);
        let right = handle_accumulation(
            &mut self.errors,
            i.right.span(),
            "expr_assign",
            &mut self.current_expr
        );
        let right = if let Some(right) = right {
            right
        } else {
            return;
        };
        new_assign.right = Box::new(right);
        self.current_expr = Some(syn::Expr::Assign(new_assign));
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);

        let assign_block = self.new_block(BlockType::ClosureAssign);
        let assign_block_id = BlockId::new(assign_block);
        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let assign_ident = syn::Ident::new(
            &format!("__closure_assign_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        if let Ok(stmt) = syn::parse2(quote::quote! { let #assign_ident; }) {
            self.cfg.graph[assign_block].statements.push(CustomStmt { stmt });
        } else {
            self.errors.push(
                Error::SynParseError(
                    closure.span(),
                    "CFG builder",
                    "expr_closure::assign_stmt".to_string()
                )
            );
            return;
        }

        let init_block = self.new_block(BlockType::ClosureArgs);
        let init_block_id = BlockId::new(init_block);
        self.connect_to(init_block);
        self.set_current_block(init_block);
        for arg in closure.inputs.iter() {
            self.visit_pat(arg);
            let pat = handle_accumulation(
                &mut self.errors,
                arg.span(),
                "expr_closure",
                &mut self.current_pat
            );
            let pat = if let Some(pat) = pat {
                pat
            } else {
                return;
            };
            let pat_stmt = syn::parse2(quote::quote! { let #pat; }).expect("pat stmt is none");
            self.push_stmt(pat_stmt);
        }

        let body_block = self.new_block(BlockType::ClosureBody);
        let body_block_id = BlockId::new(body_block);
        self.connect_to(body_block);
        self.set_current_block(body_block);
        self.set_current_block_id(body_block_id);
        self.visit_expr(&closure.body);
        let body = handle_accumulation(
            &mut self.errors,
            closure.body.span(),
            "expr_closure",
            &mut self.current_expr
        );
        let body = if let Some(body) = body {
            body
        } else {
            return;
        };
        let body_stmt = syn::Stmt::Expr(body, None);
        self.push_stmt(body_stmt);
        let body_block_id = core::mem::take(&mut self.block_ids);
        let after_block = self.new_block(BlockType::Normal);
        let after_block_id = BlockId::new(after_block);
        self.connect_to(after_block);
        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(init_block_id);
        current_block_id.children.push(body_block_id);
        current_block_id.children.push(after_block_id);
        self.set_current_block_id(current_block_id);
        self.set_current_block(after_block);

        self.current_expr = Some(
            syn::parse2(quote::quote! { #assign_ident }).expect("expr is none")
        );
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.current_item = Some(syn::Item::Const(i.clone()));
    }

    fn visit_expr(&mut self, node: &'ast syn::Expr) {
        match node {
            syn::Expr::Array(expr_array) => self.visit_expr_array(expr_array),
            syn::Expr::Assign(expr_assign) => self.visit_expr_assign(expr_assign),
            syn::Expr::Async(expr_async) => self.visit_expr_async(expr_async),
            syn::Expr::Await(expr_await) => self.visit_expr_await(expr_await),
            syn::Expr::Binary(expr_binary) => self.visit_expr_binary(expr_binary),
            syn::Expr::Block(expr_block) => self.visit_expr_block(expr_block),
            syn::Expr::Break(expr_break) => self.visit_expr_break(expr_break),
            syn::Expr::Call(expr_call) => self.visit_expr_call(expr_call),
            syn::Expr::Cast(expr_cast) => self.visit_expr_cast(expr_cast),
            syn::Expr::Closure(expr_closure) => self.visit_expr_closure(expr_closure),
            syn::Expr::Const(expr_const) => self.visit_expr_const(expr_const),
            syn::Expr::Continue(expr_continue) => self.visit_expr_continue(expr_continue),
            syn::Expr::Field(expr_field) => self.visit_expr_field(expr_field),
            syn::Expr::ForLoop(expr_for_loop) => self.visit_expr_for_loop(expr_for_loop),
            syn::Expr::Group(expr_group) => self.visit_expr_group(expr_group),
            syn::Expr::If(expr_if) => self.visit_expr_if(expr_if),
            syn::Expr::Index(expr_index) => self.visit_expr_index(expr_index),
            syn::Expr::Infer(expr_infer) => self.visit_expr_infer(expr_infer),
            syn::Expr::Let(expr_let) => self.visit_expr_let(expr_let),
            syn::Expr::Lit(expr_lit) => self.visit_expr_lit(expr_lit),
            syn::Expr::Loop(expr_loop) => self.visit_expr_loop(expr_loop),
            syn::Expr::Macro(expr_macro) => self.visit_expr_macro(expr_macro),
            syn::Expr::Match(expr_match) => self.visit_expr_match(expr_match),
            syn::Expr::MethodCall(expr_method_call) =>
                self.visit_expr_method_call(expr_method_call),
            syn::Expr::Paren(expr_paren) => self.visit_expr_paren(expr_paren),
            syn::Expr::Path(expr_path) => self.visit_expr_path(expr_path),
            syn::Expr::Range(expr_range) => self.visit_expr_range(expr_range),
            syn::Expr::RawAddr(expr_raw_addr) => self.visit_expr_raw_addr(expr_raw_addr),
            syn::Expr::Reference(expr_reference) => self.visit_expr_reference(expr_reference),
            syn::Expr::Repeat(expr_repeat) => self.visit_expr_repeat(expr_repeat),
            syn::Expr::Return(expr_return) => self.visit_expr_return(expr_return),
            syn::Expr::Struct(expr_struct) => self.visit_expr_struct(expr_struct),
            syn::Expr::Try(expr_try) => self.visit_expr_try(expr_try),
            syn::Expr::TryBlock(expr_try_block) => self.visit_expr_try_block(expr_try_block),
            syn::Expr::Tuple(expr_tuple) => self.visit_expr_tuple(expr_tuple),
            syn::Expr::Unary(expr_unary) => self.visit_expr_unary(expr_unary),
            syn::Expr::Unsafe(expr_unsafe) => self.visit_expr_unsafe(expr_unsafe),
            syn::Expr::Verbatim(_) => unimplemented!("cfg_builder::visit_expr_verbatim"),
            syn::Expr::While(expr_while) => self.visit_expr_while(expr_while),
            syn::Expr::Yield(expr_yield) => self.visit_expr_yield(expr_yield),
            _ => todo!(),
        }
    }

    fn visit_block(&mut self, node: &'ast syn::Block) {
        for (idx, it) in node.stmts.iter().enumerate() {
            self.is_last_stmt = idx == node.stmts.len() - 1;
            self.visit_stmt(it);
        }
        self.is_last_stmt = false;
    }

    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        match node {
            syn::Stmt::Local(local) => {
                self.has_assignment = true;
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    let expr = handle_accumulation(
                        &mut self.errors,
                        init.expr.span(),
                        "stmt_local",
                        &mut self.current_expr
                    );
                    let expr = if let Some(expr) = expr {
                        expr
                    } else {
                        return;
                    };
                    let mut new_local = local.clone();
                    new_local.init.as_mut().unwrap().expr = Box::new(expr);
                    self.push_stmt(syn::Stmt::Local(new_local));
                } else {
                    if let Ok(stmt) = syn::parse2(quote::quote! { #local }) {
                        self.push_stmt(stmt);
                    } else {
                        self.errors.push(
                            Error::SynParseError(
                                local.span(),
                                "CFG builder",
                                "stmt_local::stmt".to_string()
                            )
                        );
                    }
                }
                self.has_assignment = false;
            }
            syn::Stmt::Item(item) => {
                if let syn::Item::Macro(_) = item {
                    self.push_stmt(syn::Stmt::Item(item.clone()));
                } else {
                    self.visit_item(item);
                }
                match item {
                    syn::Item::ExternCrate(_) => todo!(),
                    | syn::Item::Fn(_)
                    | syn::Item::Enum(_)
                    | syn::Item::Macro(_)
                    | syn::Item::Trait(_)
                    | syn::Item::Struct(_) => {}
                    syn::Item::ForeignMod(_) => todo!(),
                    syn::Item::Impl(_) => todo!(),
                    syn::Item::Mod(_) => todo!(),
                    syn::Item::Static(_) => todo!(),
                    syn::Item::TraitAlias(_) => todo!(),
                    syn::Item::Type(_) => todo!(),
                    syn::Item::Union(_) => todo!(),
                    syn::Item::Use(_) => todo!(),
                    syn::Item::Verbatim(_) => todo!(),
                    _ => {
                        let item = handle_accumulation(
                            &mut self.errors,
                            item.span(),
                            "stmt_item",
                            &mut self.current_item
                        );
                        let item = if let Some(item) = item {
                            item
                        } else {
                            return;
                        };
                        self.push_stmt(syn::Stmt::Item(item));
                    }
                }
            }
            syn::Stmt::Expr(expr, semi) => {
                let is_last_stmt = self.is_last_stmt;
                self.visit_expr(expr);
                let new_expr = handle_accumulation(
                    &mut self.errors,
                    expr.span(),
                    "stmt_expr",
                    &mut self.current_expr
                );
                let new_expr = if let Some(new_expr) = new_expr {
                    new_expr
                } else {
                    return;
                };
                match new_expr {
                    syn::Expr::Path(_) => {
                        if is_last_stmt {
                            self.push_stmt(syn::Stmt::Expr(new_expr, semi.clone()));
                        }
                    }
                    _ => {
                        self.push_stmt(syn::Stmt::Expr(new_expr, semi.clone()));
                    }
                }
            }
            syn::Stmt::Macro(_) => {}
        }
    }
}

fn handle_accumulation<T>(
    errors: &mut Vec<Error>,
    span: proc_macro2::Span,
    msg: &str,
    current_expr: &mut Option<T>
) -> Option<T> {
    if let Some(left) = core::mem::take(current_expr) {
        Some(left)
    } else {
        errors.push(Error::ExprAccumulateError(span, "CFG builder", msg.to_string()));
        None
    }
}
