use syn::{ visit::Visit, ExprBinary, ExprMethodCall };

pub enum Operations<'ast> {
    ExprBinary(&'ast ExprBinary),
    Call(&'ast ExprMethodCall),
}

struct Graph<'ast> {
    operations: Vec<Operations<'ast>>,
}

struct GraphVisitor<'ast> {
    graphs: Vec<Graph<'ast>>,
}

impl<'ast> Visit<'ast> for GraphVisitor<'ast> {
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if let Some(graph) = self.graphs.last_mut() {
            graph.operations.push(Operations::Call(node));
        } else {
            let mut graph = Graph { operations: vec![] };
            graph.operations.push(Operations::Call(node));
            self.graphs.push(graph);
        }

        // 访问接收者和参数
        self.visit_expr(&node.receiver);
        for arg in &node.args {
            self.visit_expr(arg);
        }
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        self.visit_expr(&i.left);
        self.visit_expr(&i.right);

        if let Some(graph) = self.graphs.last_mut() {
            graph.operations.push(Operations::ExprBinary(i));
        } else {
            let mut graph = Graph { operations: vec![] };
            graph.operations.push(Operations::ExprBinary(i));
            self.graphs.push(graph);
        }
    }
}
