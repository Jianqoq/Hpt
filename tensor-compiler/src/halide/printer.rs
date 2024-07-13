use tensor_types::dtype::Dtype;

use crate::halide::exprs::Int;

use super::{ prime_expr::PrimeExpr, stmt::Stmt };

pub struct IRPrinter;

impl IRPrinter {
    pub fn print_stmt<T: Into<Stmt>>(&self, stmt: T) {
        _IRPrinter::new(0).print_stmt(stmt);
    }

    pub fn print_stmt_str<T: Into<Stmt>>(&self, stmt: T) -> String {
        _IRPrinter::new(0).print_stmt_str(stmt)
    }

    pub fn print_expr<T: Into<PrimeExpr>>(&mut self, expr: T) {
        let expr = expr.into();
        println!("{}", expr);
    }
}

pub(crate) struct _IRPrinter {
    indent: usize,
}

impl _IRPrinter {
    pub fn new(indent: usize) -> Self {
        _IRPrinter { indent }
    }

    fn do_indent(&self) {
        print!("{}", "    ".repeat(self.indent));
    }
    fn do_indent_str(&self) -> String {
        "    ".repeat(self.indent)
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
                if var.step() == &Int::make(Dtype::I64, 1).into() {
                    println!("for {} in range({}, {}) {{", var.var(), var.start(), var.end());
                } else {
                    println!(
                        "for {} in range({}, {}, {}) {{",
                        var.var(),
                        var.start(),
                        var.end(),
                        var.step()
                    );
                }
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
            Stmt::Return(var) => {
                self.do_indent();
                println!("{}", var);
            }
            Stmt::None => {}
        }
    }

    pub fn print_stmt_str<T: Into<Stmt>>(&mut self, stmt: T) -> String {
        let stmt = stmt.into();
        let mut res = String::new();
        match stmt {
            Stmt::LetStmt(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("let {} = {};\n", var.var(), var.body()));
            }
            Stmt::StoreStmt(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::AssignStmt(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{} = {};\n", var.lhs(), var.rhs()));
            }
            Stmt::For(var) => {
                res.push_str(&self.do_indent_str());
                if var.step() == &Int::make(Dtype::I64, 1).into() {
                    res.push_str(
                        &format!("for {} in range({}, {}) {{\n", var.var(), var.start(), var.end())
                    );
                } else {
                    res.push_str(
                        &format!(
                            "for {} in range({}, {}, {}) {{\n",
                            var.var(),
                            var.start(),
                            var.end(),
                            var.step()
                        )
                    );
                }
                self.indent += 1;
                res.push_str(&self.print_stmt_str(var.stmt()));
                self.indent -= 1;
                res.push_str(&self.do_indent_str());
                res.push_str("}\n");
            }
            Stmt::Seq(stmts) => {
                for stmt in stmts.stmts() {
                    res.push_str(&self.print_stmt_str(stmt));
                }
            }
            Stmt::IfThenElse(stmt) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("if {} {{\n", stmt.cond()));
                self.indent += 1;
                res.push_str(&self.print_stmt_str(stmt.then_case()));
                self.indent -= 1;
                res.push_str(&self.do_indent_str());
                let else_case = stmt.else_case();
                if else_case.is_none() {
                    res.push_str("}\n");
                    return res;
                } else {
                    res.push_str("} else {\n");
                    self.indent += 1;
                    res.push_str(&self.print_stmt_str(stmt.else_case()));
                    self.indent -= 1;
                    res.push_str(&self.do_indent_str());
                    res.push_str("}\n");
                }
            }
            Stmt::InplaceStore(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::InplaceAdd(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::InplaceSub(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::InplaceMul(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::InplaceDiv(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::Return(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!("{}\n", var));
            }
            Stmt::None => {}
        }
        res
    }
}
