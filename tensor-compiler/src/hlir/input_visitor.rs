use hashbrown::HashSet;

use super::{ expr::Expr, tensor::Tensor, traits::HlirMutVisitor };

pub struct InputVisitor {
    inputs: HashSet<Tensor>,
}

impl InputVisitor {
    pub fn new() -> Self {
        InputVisitor {
            inputs: HashSet::new(),
        }
    }
    pub fn inputs(&self) -> &HashSet<Tensor> {
        &self.inputs
    }
    pub fn visit(expr: &Expr) -> HashSet<Tensor> {
        let mut visitor = InputVisitor::new();
        visitor.visit_expr(expr);
        visitor.inputs
    }
}

impl HlirMutVisitor for InputVisitor {
    fn visit_tensor(&mut self, t: &Tensor) {
        self.inputs.insert(t.clone());
    }
}
