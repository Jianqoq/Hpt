pub(crate) struct ExprExpander {
    pub(crate) stmts: Vec<syn::Stmt>,
    pub(crate) current_expr: Option<syn::Expr>,
}

impl ExprExpander {
    pub(crate) fn new() -> Self {
        Self { stmts: Vec::new(), current_expr: None }
    }
}

impl<'ast> syn::visit::Visit<'ast> for ExprExpander {
    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        match stmt {
            syn::Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    if let Some(expr) = self.current_expr.take() {
                        let mut new_local = local.clone();
                        new_local.init.as_mut().unwrap().expr = Box::new(expr);
                        self.stmts.push(syn::Stmt::Local(new_local));
                    } else {
                        self.stmts.push(stmt.clone());
                    }
                } else {
                    self.stmts.push(stmt.clone());
                }
            }
            syn::Stmt::Expr(expr, _) => {
                self.visit_expr(expr);
            }
            _ => {
                self.stmts.push(stmt.clone());
            }
        }
    }

    fn visit_expr(&mut self, expr: &'ast syn::Expr) {
        match expr {
            syn::Expr::Array(_) => unimplemented!("expr_expand::visit_expr::Array"),
            syn::Expr::Assign(_) => unimplemented!("expr_expand::visit_expr::Assign"),
            syn::Expr::Async(_) => unimplemented!("expr_expand::visit_expr::Async"),
            syn::Expr::Await(_) => unimplemented!("expr_expand::visit_expr::Await"),
            syn::Expr::Binary(expr_binary) => self.visit_expr_binary(expr_binary),
            syn::Expr::Block(_) => unimplemented!("expr_expand::visit_expr::Block"),
            syn::Expr::Break(_) => unimplemented!("expr_expand::visit_expr::Break"),
            syn::Expr::Call(call) => {
                for arg in call.args.iter() {
                    // self.visit_expr(arg);
                    // if let Some(expr) = self.current_expr.take() {
                    //     if let syn::Expr::Path(path) = expr {

                    //     } else {

                    //     }
                    // }
                }
            }
            syn::Expr::Cast(_) => unimplemented!("expr_expand::visit_expr::Cast"),
            syn::Expr::Closure(_) => unimplemented!("expr_expand::visit_expr::Closure"),
            syn::Expr::Const(_) => unimplemented!("expr_expand::visit_expr::Const"),
            syn::Expr::Continue(_) => unimplemented!("expr_expand::visit_expr::Continue"),
            syn::Expr::Field(_) => unimplemented!("expr_expand::visit_expr::Field"),
            syn::Expr::Group(_) => unimplemented!("expr_expand::visit_expr::Group"),
            syn::Expr::Index(_) => unimplemented!("expr_expand::visit_expr::Index"),
            syn::Expr::Infer(_) => unimplemented!("expr_expand::visit_expr::Infer"),
            syn::Expr::Let(_) => unimplemented!("expr_expand::visit_expr::Let"),
            syn::Expr::Lit(_) => unimplemented!("expr_expand::visit_expr::Lit"),
            syn::Expr::Macro(_) => unimplemented!("expr_expand::visit_expr::Macro"),
            syn::Expr::Match(_) => unimplemented!("expr_expand::visit_expr::Match"),
            syn::Expr::MethodCall(_) => unimplemented!("expr_expand::visit_expr::MethodCall"),
            syn::Expr::Paren(_) => unimplemented!("expr_expand::visit_expr::Paren"),
            syn::Expr::Range(_) => unimplemented!("expr_expand::visit_expr::Range"),
            syn::Expr::RawAddr(_) => unimplemented!("expr_expand::visit_expr::RawAddr"),
            syn::Expr::Repeat(_) => unimplemented!("expr_expand::visit_expr::Repeat"),
            syn::Expr::Return(_) => unimplemented!("expr_expand::visit_expr::Return"),
            syn::Expr::Struct(_) => unimplemented!("expr_expand::visit_expr::Struct"),
            syn::Expr::Try(_) => unimplemented!("expr_expand::visit_expr::Try"),
            syn::Expr::TryBlock(_) => unimplemented!("expr_expand::visit_expr::TryBlock"),
            syn::Expr::Tuple(_) => unimplemented!("expr_expand::visit_expr::Tuple"),
            syn::Expr::Unary(_) => unimplemented!("expr_expand::visit_expr::Unary"),
            syn::Expr::Unsafe(_) => unimplemented!("expr_expand::visit_expr::Unsafe"),
            syn::Expr::Verbatim(_) => unimplemented!("expr_expand::visit_expr::Verbatim"),
            syn::Expr::Yield(_) => unimplemented!("expr_expand::visit_expr::Yield"),
            _ => syn::visit::visit_expr(self, expr),
        }
    }

    fn visit_expr_reference(&mut self, reference: &'ast syn::ExprReference) {
        self.visit_expr(&reference.expr);
        if let Some(expr) = self.current_expr.take() {
            let mut new_reference = reference.clone();
            new_reference.expr = Box::new(expr);
            self.current_expr = Some(syn::Expr::Reference(new_reference));
        } else {
            self.current_expr = Some(syn::Expr::Reference(reference.clone()));
        }
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.current_expr = Some(syn::Expr::Path(path.clone()));
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        match (i.left.as_ref(), i.right.as_ref()) {
            (syn::Expr::Path(left), syn::Expr::Binary(_) | syn::Expr::Reference(_)) => {
                self.visit_expr(i.right.as_ref());
                let right_expr = self.current_expr.take().unwrap();
                let right_local = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote!(
                    let __out2 = #right_expr;
                )
                    )
                    .expect("failed to parse stmt::115");
                self.stmts.push(right_local);
                let op = &i.op;
                self.current_expr = Some(
                    syn
                        ::parse2::<syn::Expr>(quote::quote!(#left #op __out2))
                        .expect("failed to parse expr::119")
                );
            }
            (
                syn::Expr::Binary(_) | syn::Expr::Reference(_),
                syn::Expr::Binary(_) | syn::Expr::Reference(_),
            ) => {
                self.visit_expr(i.left.as_ref());
                let left_expr = self.current_expr.take().unwrap();
                let left_local = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote!(
                    let __out1 = #left_expr;
                )
                    )
                    .expect("failed to parse stmt::134");
                self.stmts.push(left_local);
                self.visit_expr(i.right.as_ref());
                let right_expr = self.current_expr.take().unwrap();
                let right_local = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote!(
                    let __out2 = #right_expr;
                )
                    )
                    .expect("failed to parse stmt::144");
                self.stmts.push(right_local);
                let op = &i.op;
                self.current_expr = Some(
                    syn
                        ::parse2::<syn::Expr>(quote::quote!(__out1 #op __out2))
                        .expect("failed to parse expr::148")
                );
            }
            (syn::Expr::Binary(_) | syn::Expr::Reference(_), syn::Expr::Path(right)) => {
                self.visit_expr(i.left.as_ref());
                let left_expr = self.current_expr.take().unwrap();
                let left_local = syn
                    ::parse2::<syn::Stmt>(
                        quote::quote!(
                    let __out1 = #left_expr;
                )
                    )
                    .expect("failed to parse stmt::160");
                self.stmts.push(left_local);
                let op = &i.op;
                self.current_expr = Some(
                    syn
                        ::parse2::<syn::Expr>(quote::quote!(__out1 #op #right))
                        .expect("failed to parse expr::164")
                );
            }
            _ => {
                self.current_expr = Some(syn::Expr::Binary(i.clone()));
            }
        }
    }
}
