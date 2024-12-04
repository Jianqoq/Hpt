pub(crate) struct ControlFlowDetector {
    pub(crate) has_control_flow: bool,
}

impl ControlFlowDetector {
    pub(crate) fn new() -> Self {
        Self { has_control_flow: false }
    }
}

impl<'ast> syn::visit::Visit<'ast> for ControlFlowDetector {
    fn visit_expr_for_loop(&mut self, _: &'ast syn::ExprForLoop) {
        self.has_control_flow = true;
    }
    fn visit_expr_if(&mut self, _: &'ast syn::ExprIf) {
        self.has_control_flow = true;
    }
    fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {
        self.has_control_flow = true;
    }
    fn visit_expr_match(&mut self, _: &'ast syn::ExprMatch) {
        self.has_control_flow = true;
    }
    fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {
        self.has_control_flow = true;
    }
    fn visit_expr_block(&mut self, _: &'ast syn::ExprBlock) {
        self.has_control_flow = true;
    }
    fn visit_block(&mut self, _: &'ast syn::Block) {
        self.has_control_flow = true;
    }
}
