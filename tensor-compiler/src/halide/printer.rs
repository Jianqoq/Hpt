use colored::Colorize;

use tensor_types::dtype::Dtype;

use crate::halide::exprs::Int;

use super::{module::Module, prime_expr::PrimeExpr, stmt::Stmt};

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

    pub fn print_module(&mut self, module: &Module) {
        _IRPrinter::new(0).print_module(module);
    }

    pub fn print_module_str(&mut self, module: &Module) -> String {
        _IRPrinter::new(0).print_module_str(module)
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

    pub fn print_module(&mut self, module: &Module) {
        println!("module {} {{", module.name);
        self.indent += 1;
        for import in &module.imports {
            println!("import {};", import);
        }
        for fn_meta in module.fns.values() {
            self.do_indent();
            print!("fn {}(", fn_meta.function.name);
            for (i, (name, r#type)) in fn_meta.function.ty.args.iter().enumerate() {
                if i != 0 {
                    print!(", ");
                }
                print!("{}: {}", name, r#type);
            }
            println!(") -> {} {{", fn_meta.function.ty.ret_ty);
            _IRPrinter::new(self.indent + 1).print_stmt(&fn_meta.function.body);
            self.do_indent();
            println!("}}");
        }
        println!("}}");
    }

    pub fn print_module_str(&mut self, module: &Module) -> String {
        let mut res = String::new();
        res.push_str(&format!("module {} {{\n", module.name));
        self.indent += 1;
        for import in &module.imports {
            res.push_str(&format!("import {};\n", import));
        }
        for fn_meta in module.fns.values() {
            res.push_str(&self.do_indent_str());
            res.push_str(&format!("fn {}(", fn_meta.function.name));
            for (i, (name, r#type)) in fn_meta.function.ty.args.iter().enumerate() {
                if i != 0 {
                    res.push_str(", ");
                }
                res.push_str(&format!("{}: {}", name, r#type));
            }
            res.push_str(&format!(") -> {} {{\n", fn_meta.function.ty.ret_ty));
            res.push_str(&format!(
                "{}",
                _IRPrinter::new(self.indent + 1).print_stmt_str(&fn_meta.function.body)
            ));
            res.push_str(&self.do_indent_str());
            res.push_str("}\n");
        }
        res.push_str("}");
        res
    }

    pub fn print_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        let stmt = stmt.into();
        match stmt {
            Stmt::LetStmt(var) => {
                self.do_indent();
                println!("let {} = {};", var.var(), var.value());
                self.print_stmt(var.body());
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
                    println!(
                        "for {} in range({}, {}) {{",
                        var.var(),
                        var.start(),
                        var.end()
                    );
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
            Stmt::AllocaStmt(var) => {
                self.do_indent();
                println!(
                    "let {} = alloca<{}>({});",
                    var.var(),
                    var.dtype(),
                    var.size()
                );
                self.print_stmt(var.body());
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
                res.push_str(&format!(
                    "{} {} {} {};\n",
                    "let".purple(),
                    var.var(),
                    "=".purple(),
                    var.value()
                ));
                res.push_str(&self.print_stmt_str(var.body()));
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
                    res.push_str(&format!(
                        "{} {} {} {}{}{}, {}{} {{\n",
                        "for".purple(),
                        var.var(),
                        "in".purple(),
                        "range".blue(),
                        "(".bright_cyan(),
                        var.start(),
                        var.end(),
                        ")".bright_cyan()
                    ));
                } else {
                    res.push_str(&format!(
                        "{} {} {} {}{}{}, {}, {}{} {{\n",
                        "for".purple(),
                        var.var(),
                        "in".purple(),
                        "range".blue(),
                        "(".bright_cyan(),
                        var.start(),
                        var.end(),
                        var.step(),
                        ")".bright_cyan()
                    ));
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
                res.push_str(&format!("{} {} {{\n", "if".purple(), stmt.cond()));
                self.indent += 1;
                res.push_str(&self.print_stmt_str(stmt.then_case()));
                self.indent -= 1;
                res.push_str(&self.do_indent_str());
                let else_case = stmt.else_case();
                if else_case.is_none() {
                    res.push_str("}\n");
                    return res;
                } else {
                    res.push_str(&format!("}} {} {{\n", "else".purple()));
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
            Stmt::AllocaStmt(var) => {
                res.push_str(&self.do_indent_str());
                res.push_str(&format!(
                    "{} {} {} {}<{}>{}{}{};\n",
                    "let".purple(),
                    var.var(),
                    "=".purple(),
                    "alloc".blue(),
                    var.dtype(),
                    "(".bright_cyan(),
                    var.size(),
                    ")".bright_cyan()
                ));
                res.push_str(&self.print_stmt_str(var.body()));
            }
            Stmt::None => {}
        }
        res
    }
}
