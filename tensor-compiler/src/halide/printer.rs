use super::{ prime_expr::PrimeExpr, stmt::Stmt };

pub struct IRPrinter;

impl IRPrinter {
    pub fn print_stmt<T: Into<Stmt>>(&self, stmt: T) {
        _IRPrinter::new().print_stmt(stmt);
    }

    pub fn print_expr<T: Into<PrimeExpr>>(&mut self, expr: T) {
        let expr = expr.into();
        println!("{}", expr);
    }
}

struct _IRPrinter {
    indent: usize,
}

impl _IRPrinter {
    pub fn new() -> Self {
        _IRPrinter { indent: 0 }
    }

    fn do_indent(&self) {
        print!("{}", "    ".repeat(self.indent));
    }

    pub fn print_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        let stmt = stmt.into();
        match stmt {
            Stmt::LetStmt(var) => {
                self.do_indent();
                println!("let {} = {};", var.var(), var.body());
            }
            Stmt::StoreStmt(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::AssignStmt(var) => {
                self.do_indent();
                println!("{} = {};", var.lhs(), var.rhs());
            }
            Stmt::For(var) => {
                self.do_indent();
                println!("for {} in range({}, {}) {{", var.var(), var.start(), var.end());
                self.indent += 1;
                self.print_stmt(var.stmt());
                self.indent -= 1;
                self.do_indent();
                println!("}}");
            }
            Stmt::Seq(stmts) => {
                for stmt in stmts.stmts() {
                    self.print_stmt(stmt);
                }
            }
            Stmt::IfThenElse(stmt) => {
                self.do_indent();
                println!("if {} {{", stmt.cond());
                self.indent += 1;
                self.print_stmt(stmt.then_case());
                self.indent -= 1;
                self.do_indent();
                let else_case = stmt.else_case();
                if else_case.is_none() {
                    println!("}}");
                    return;
                } else {
                    println!("}} else {{");
                    self.indent += 1;
                    self.print_stmt(stmt.else_case());
                    self.indent -= 1;
                    self.do_indent();
                    println!("}}");
                }
            }
            Stmt::InplaceStore(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::InplaceAdd(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::InplaceSub(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::InplaceMul(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::InplaceDiv(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::None => {}
        }
    }
}
