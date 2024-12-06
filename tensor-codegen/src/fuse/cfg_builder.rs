use std::collections::{ HashMap, HashSet };

use petgraph::graph::NodeIndex;
use syn::visit::Visit;

use super::cfg::{ BasicBlock, BlockId, BlockType, CustomStmt, CFG };

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
        let mut current_block_id = Some(core::mem::take(&mut self.block_ids));
        let (cond_block, cond_block_id, assign_ident, assign_block_id) = if
            self.cfg.graph[self.current_block].block_type == BlockType::IfElseEnd
        {
            self.cfg.graph[self.current_block].block_type = BlockType::ElseIfCond;
            (self.current_block, None, None, None)
        } else {
            let assign_block = self.new_block(BlockType::IfAssign);
            let assign_block_id = BlockId::new(assign_block);
            self.connect_to(assign_block);
            self.set_current_block(assign_block);
            let assign_ident = syn::Ident::new(
                &format!("__if_assign_{}", self.global_block_cnt()),
                proc_macro2::Span::call_site()
            );
            self.cfg.graph[assign_block].statements.push(CustomStmt {
                stmt: syn
                    ::parse2(quote::quote! { let #assign_ident; })
                    .expect("assign stmt is none"),
            });
            let new_cond_block = self.new_block(BlockType::IfCond);
            let cond_block_id = BlockId::new(new_cond_block);
            (new_cond_block, Some(cond_block_id), Some(assign_ident), Some(assign_block_id))
        };

        // 创建 then 分支块
        let then_block = self.new_block(BlockType::IfThenEnd);
        let then_block_id = BlockId::new(then_block);
        // 创建 else 分支块
        let else_block = self.new_block(BlockType::IfElseEnd);
        let else_block_id = BlockId::new(else_block);
        // 创建合并块
        let merge_block = self.new_block(BlockType::Normal);
        let merge_block_id = BlockId::new(merge_block);

        // 连接当前块到条件检查块
        self.connect_to(cond_block);
        self.set_current_block(cond_block);
        if let Some(cond_block_id) = cond_block_id {
            self.set_current_block_id(cond_block_id);
        } else {
            self.set_current_block_id(current_block_id.unwrap());
            current_block_id = None;
        }

        self.visit_expr(&expr_if.cond);
        let cond_expr = core::mem::take(&mut self.current_expr).expect("cond expr is none");
        self.push_stmt(syn::Stmt::Expr(cond_expr, None));
        let mut cond_block_id = core::mem::take(&mut self.block_ids);
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
            self.cfg.graph[then_block].block_type = BlockType::IfThen;
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

        if let Some(mut current_block_id) = current_block_id {
            if let Some(assign_block_id) = assign_block_id {
                current_block_id.children.push(assign_block_id);
            }
            current_block_id.children.push(cond_block_id);
            current_block_id.children.push(then_block_id);
            current_block_id.children.push(else_block_id);
            current_block_id.children.push(merge_block_id);
            self.set_current_block(merge_block);
            self.set_current_block_id(current_block_id);
        } else {
            if let Some(assign_block_id) = assign_block_id {
                cond_block_id.children.push(assign_block_id);
            }
            cond_block_id.children.push(then_block_id);
            cond_block_id.children.push(else_block_id);
            cond_block_id.children.push(merge_block_id);
            self.set_current_block(merge_block);
            self.set_current_block_id(cond_block_id);
        }
        // 更新当前块为合并块

        if let Some(assign_ident) = assign_ident {
            self.current_expr = Some(
                syn::parse2(quote::quote! { #assign_ident }).expect("expr is none")
            );
        }
    }

    // 处理 loop 语句
    fn handle_loop(&mut self, expr_loop: &syn::ExprLoop) {
        let mut current_block_id = core::mem::take(&mut self.block_ids);
        let assign_block = self.new_block(BlockType::LoopAssign);
        let assign_block_id = BlockId::new(assign_block);
        // 创建循环体块
        let loop_block = self.new_block(BlockType::LoopBody);
        let loop_block_id = BlockId::new(loop_block);
        // 创建循环后的块
        let after_loop_block = self.new_block(BlockType::Normal);
        let after_loop_block_id = BlockId::new(after_loop_block);

        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let assign_ident = syn::Ident::new(
            &format!("__loop_assign_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        self.cfg.graph[assign_block].statements.push(CustomStmt {
            stmt: syn::parse2(quote::quote! { let #assign_ident; }).expect("assign stmt is none"),
        });

        // 连接当前块到循环体块
        self.connect_to(loop_block);

        // 设置当前块为循环体块，并处理循环体
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_loop.body);
        self.current_expr = Some(
            syn::parse2(quote::quote! { #assign_ident }).expect("expr is none")
        );
        let loop_block_id = core::mem::take(&mut self.block_ids);

        // 连接循环体块回到自身，表示下一次迭代
        self.connect_to(loop_block);
        // 连接循环体块到循环后的块，表示退出循环
        self.connect_to(after_loop_block);

        // 更新当前块为循环后的块，继续处理后续语句
        self.set_current_block(after_loop_block);
        current_block_id.children.push(assign_block_id);
        current_block_id.children.push(loop_block_id);
        current_block_id.children.push(after_loop_block_id);
        self.set_current_block_id(current_block_id);
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
            &format!("__block_out_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
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
        self.current_expr = Some(
            syn::parse2(quote::quote! { #block_ident }).expect("block expr is none")
        );
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
        self.cfg.graph[assign_block].statements.push(CustomStmt {
            stmt: syn
                ::parse2(quote::quote! {
                let #block_ident;
            })
                .expect("cfg_builder::handle_while_loop::assign_block"),
        });

        // 连接当前块到条件检查块
        self.connect_to(condition_block);

        // 连接条件检查块到循环体块（如果条件为真）和循环后的块（如果条件为假）
        self.set_current_block(condition_block);
        self.set_current_block_id(condition_block_id);
        self.visit_expr(&expr_while.cond);
        let expr = core::mem::take(&mut self.current_expr).expect("cfg::handle_while_loop::expr");
        self.push_stmt(syn::Stmt::Expr(expr, None));
        let condition_block_id = core::mem::take(&mut self.block_ids);

        // 创建两个新的连接：条件为真和条件为假
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_while.body);
        self.current_expr = Some(
            syn::parse2(quote::quote! { #block_ident }).expect("while loop expr is none")
        );
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

        self.connect_to(assign_block);
        self.set_current_block(assign_block);
        let block_ident = syn::Ident::new(
            &format!("__for_out_{}", self.global_block_cnt()),
            proc_macro2::Span::call_site()
        );
        self.cfg.graph[assign_block].statements.push(CustomStmt {
            stmt: syn
                ::parse2(quote::quote! {
                let #block_ident;
            })
                .expect("cfg_builder::handle_for_loop::assign_block"),
        });

        // 连接当前块到初始化块
        self.connect_to(init_block);

        // 处理迭代器初始化和元素绑定
        self.set_current_block(init_block);
        self.set_current_block_id(init_block_id);
        let init_block_id = core::mem::take(&mut self.block_ids);
        let pat = &expr_for.pat;
        let local = quote::quote! {
            let #pat;
        };
        let stmt = syn::parse2(local).expect("cfg::handle_for_loop::local");
        self.push_stmt(stmt);
        // 连接初始化块到条件检查块
        self.connect_to(condition_block);

        // 处理条件检查
        self.set_current_block(condition_block);
        self.set_current_block_id(condition_block_id);
        let condition_block_id = core::mem::take(&mut self.block_ids);
        self.visit_expr(&expr_for.expr);
        let expr = core::mem::take(&mut self.current_expr).expect("cfg::handle_for_loop::expr");
        self.push_stmt(syn::Stmt::Expr(expr, None));
        // 创建连接：条件为真进入循环体块，条件为假进入循环后的块
        self.connect_to(loop_block);
        self.connect_to(after_loop_block);

        // 处理循环体
        self.set_current_block(loop_block);
        self.set_current_block_id(loop_block_id);
        self.visit_block(&expr_for.body);
        self.current_expr = Some(
            syn::parse2(quote::quote! { #block_ident }).expect("for loop expr is none")
        );
        let loop_block_id = core::mem::take(&mut self.block_ids);
        // 连接循环体块回到条件检查块（表示下一次迭代）
        self.connect_to(condition_block);
        // 连接循环体块到循环后的块（如果有 break）
        self.connect_to(after_loop_block);

        // 设置当前块为循环后的块，继续处理后续语句
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
        let (where_block, where_block_id) = if let Some(where_clause) = &i.sig.generics.where_clause {
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
        self.push_stmt(syn::parse2(quote::quote! { let #name; }).expect("fn name is none"));
        self.connect_to(args_block);
        self.set_current_block(args_block);
        for arg in &i.sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => {
                    unimplemented!("cfg_builder::visit_item_fn::receiver")
                },
                syn::FnArg::Typed(pat_type) => {
                    let arg_stmt = syn::parse2(quote::quote! { let #pat_type; }).expect("arg stmt is none::365");
                    self.push_stmt(arg_stmt);
                },
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
            syn::FnArg::Receiver(_) => { unimplemented!("cfg_builder::visit_fn_arg::receiver") }
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
        let pat = core::mem::take(&mut self.current_pat).expect("pat is none");
        self.visit_type(&i.ty);
        let ty = core::mem::take(&mut self.current_type).expect("ty is none");
        new_pat.pat = Box::new(pat);
        new_pat.ty = Box::new(ty);
        self.current_pat = Some(syn::Pat::Type(new_pat));
    }

    fn visit_expr_range(&mut self, i: &'ast syn::ExprRange) {
        let mut new_expr = i.clone();
        match (new_expr.start.as_mut(), new_expr.end.as_mut()) {
            (None, None) => {
                self.current_expr = Some(syn::Expr::Range(new_expr));
            }
            (None, Some(right)) => {
                self.visit_expr(right);
                let right = core::mem::take(&mut self.current_expr).expect("right is none");
                new_expr.end = Some(Box::new(right));
                self.current_expr = Some(syn::Expr::Range(new_expr));
            }
            (Some(left), None) => {
                self.visit_expr(left);
                let left = core::mem::take(&mut self.current_expr).expect("left is none");
                new_expr.start = Some(Box::new(left));
                self.current_expr = Some(syn::Expr::Range(new_expr));
            }
            (Some(left), Some(right)) => {
                self.visit_expr(left);
                let left = core::mem::take(&mut self.current_expr).expect("left is none");
                self.visit_expr(right);
                let right = core::mem::take(&mut self.current_expr).expect("right is none");
                new_expr.start = Some(Box::new(left));
                new_expr.end = Some(Box::new(right));
                self.current_expr = Some(syn::Expr::Range(new_expr));
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
        let expr = core::mem::take(&mut self.current_expr).expect("expr is none::374");
        new_expr.expr = Box::new(expr);
        self.current_expr = Some(syn::Expr::Reference(new_expr));
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        let mut new_binary = binary.clone();

        match (new_binary.left.as_mut(), new_binary.right.as_mut()) {
            (syn::Expr::Binary(left), syn::Expr::Binary(right)) => {
                self.visit_expr_binary(left);
                let left = core::mem::take(&mut self.current_expr).expect("left is none");
                self.visit_expr_binary(right);
                let right = core::mem::take(&mut self.current_expr).expect("right is none");
                let op = &binary.op;
                let left_stmt = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote! {
                    let __out_1 = #left;
                }
                    )
                    .unwrap();
                let right_stmt = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote! {
                    let __out_2 = #right;
                }
                    )
                    .expect("right stmt is none");
                let expr = syn
                    ::parse2::<syn::Expr>(
                        quote::quote! {
                    __out_1 #op __out_2
                }
                    )
                    .expect("expr is none::409");
                self.current_expr = Some(expr);
                self.push_stmt(left_stmt);
                self.push_stmt(right_stmt);
            }
            (syn::Expr::Binary(left), _) => {
                self.visit_expr_binary(left);
                let left = core::mem::take(&mut self.current_expr).expect("left is none");
                self.visit_expr(binary.right.as_ref());
                let right = core::mem::take(&mut self.current_expr).expect("right is none");
                let op = &binary.op;
                let left_stmt = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote! {
                    let __out_1 = #left;
                }
                    )
                    .unwrap();
                let expr = syn
                    ::parse2::<syn::Expr>(
                        quote::quote! {
                    __out_1 #op #right
                }
                    )
                    .expect("expr is none::433");
                self.current_expr = Some(expr);
                self.push_stmt(left_stmt);
            }
            (_, syn::Expr::Binary(right)) => {
                self.visit_expr(binary.left.as_ref());
                let left = core::mem::take(&mut self.current_expr).expect("left is none");
                self.visit_expr_binary(right);
                let right = core::mem::take(&mut self.current_expr).expect("right is none");
                let op = &binary.op;
                let right_stmt = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote! {
                    let __out_2 = #right;
                }
                    )
                    .expect("right stmt is none");
                let expr = syn
                    ::parse2::<syn::Expr>(
                        quote::quote! {
                    #left #op __out_2
                }
                    )
                    .expect("expr is none::456");
                self.current_expr = Some(expr);
                self.push_stmt(right_stmt);
            }
            _ => {
                self.visit_expr(binary.left.as_ref());
                let left = core::mem::take(&mut self.current_expr).expect("left is none");
                self.visit_expr(binary.right.as_ref());
                let right = core::mem::take(&mut self.current_expr).expect("right is none");
                let op = &binary.op;
                let expr = syn
                    ::parse2(quote::quote! { #left #op #right })
                    .expect("expr is none::466");
                self.current_expr = Some(expr);
            }
        }
    }

    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        self.current_expr = Some(syn::Expr::Lit(i.clone()));
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let mut new_call = call.clone();
        self.visit_expr(call.func.as_ref());
        let func = core::mem::take(&mut self.current_expr).expect("func is none");
        for arg in new_call.args.iter_mut() {
            self.visit_expr(arg);
            let new_arg = core::mem::take(&mut self.current_expr).expect("arg is none::546");
            *arg = new_arg;
        }
        new_call.func = Box::new(func);
        self.current_expr = Some(syn::Expr::Call(new_call));
    }
    fn visit_expr_method_call(&mut self, method_call: &'ast syn::ExprMethodCall) {
        let mut new_method_call = method_call.clone();
        self.visit_expr(method_call.receiver.as_ref());
        let receiver = core::mem::take(&mut self.current_expr).expect("receiver is none");
        match receiver {
            syn::Expr::Binary(_) | syn::Expr::Reference(_) | syn::Expr::Tuple(_) => {
                self.push_stmt(
                    syn
                        ::parse2(
                            quote::quote! {
                        let __expr_receiver = #receiver;
                    }
                        )
                        .expect("stmt is none")
                );
                new_method_call.receiver = Box::new(
                    syn::parse2(quote::quote! { __expr_receiver }).expect("expr is none::519")
                );
            }
            _ => {
                new_method_call.receiver = Box::new(receiver);
            }
        }
        for arg in new_method_call.args.iter_mut() {
            self.visit_expr(arg);
            let new_arg = core::mem::take(&mut self.current_expr).expect("arg is none::577");
            *arg = new_arg;
        }
        self.current_expr = Some(syn::Expr::MethodCall(new_method_call));
    }

    fn visit_expr_tuple(&mut self, tuple: &'ast syn::ExprTuple) {
        let mut new_tuple = tuple.clone();
        for elem in new_tuple.elems.iter_mut() {
            self.visit_expr(elem);
            let new_elem = core::mem::take(&mut self.current_expr).expect("elem is none");
            *elem = new_elem;
        }
        self.push_stmt(
            syn::parse2(quote::quote! { let __expr_tuple = #new_tuple; }).expect("stmt is none")
        );
        self.current_expr = Some(
            syn::parse2(quote::quote! { __expr_tuple }).expect("expr is none")
        );
    }

    fn visit_expr_try(&mut self, i: &'ast syn::ExprTry) {
        let mut new_try = i.clone();
        self.visit_expr(&i.expr);
        let expr = core::mem::take(&mut self.current_expr).expect("expr is none");
        new_try.expr = Box::new(expr);
        self.current_expr = Some(syn::Expr::Try(new_try));
    }

    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        let mut new_assign = i.clone();
        self.visit_expr(&i.right);
        let right = core::mem::take(&mut self.current_expr).expect("right is none");
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
        self.cfg.graph[assign_block].statements.push(CustomStmt {
            stmt: syn::parse2(quote::quote! { let #assign_ident; }).expect("assign stmt is none"),
        });

        let init_block = self.new_block(BlockType::ClosureArgs);
        let init_block_id = BlockId::new(init_block);
        self.connect_to(init_block);
        self.set_current_block(init_block);
        for arg in closure.inputs.iter() {
            self.visit_pat(arg);
            let pat = core::mem::take(&mut self.current_pat).expect("pat is none");
            let pat_stmt = syn::parse2(quote::quote! { let #pat; }).expect("pat stmt is none");
            self.push_stmt(pat_stmt);
        }

        let body_block = self.new_block(BlockType::ClosureBody);
        let body_block_id = BlockId::new(body_block);
        self.connect_to(body_block);
        self.set_current_block(body_block);
        self.set_current_block_id(body_block_id);
        self.visit_expr(&closure.body);
        let body = core::mem::take(&mut self.current_expr).expect("body is none");
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
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    let expr = core::mem::take(&mut self.current_expr).expect("init is none");
                    let mut new_local = local.clone();
                    new_local.init.as_mut().unwrap().expr = Box::new(expr);
                    self.push_stmt(syn::Stmt::Local(new_local));
                } else {
                    self.push_stmt(
                        syn::parse2(quote::quote! { #local }).expect("local stmt is none")
                    );
                }
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
                        let item = core::mem::take(&mut self.current_item).expect("item is none");
                        self.push_stmt(syn::Stmt::Item(item));
                    }
                }
            }
            syn::Stmt::Expr(expr, semi) => {
                let is_last_stmt = self.is_last_stmt;
                self.visit_expr(expr);
                let new_expr = core::mem::take(&mut self.current_expr).expect("expr is none::604");
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
