#![allow(dead_code)]
use super::node::Expr;

pub struct HlirPrinter;

impl HlirPrinter {
    pub fn print_expr<T: Into<Expr>>(&mut self, expr: T) {
        let expr = expr.into();
        println!("{}", expr);
    }
}

struct _HlirPrinter {
    indent: usize,
}

impl _HlirPrinter {
    pub fn new() -> Self {
        _HlirPrinter { indent: 0 }
    }

    fn do_indent(&self) {
        print!("{}", "    ".repeat(self.indent));
    }

    pub fn print<T: Into<Expr>>(&mut self, stmt: T) {
        let stmt = stmt.into();
        match stmt {
            Expr::Let(var) => {
                self.do_indent();
                println!("let {} = {};", var.var(), var.value());
            }
            Expr::For(var) => {
                self.do_indent();
                println!(
                    "for {} in range({}, {}, {}) {{",
                    var.var(),
                    var.start(),
                    var.end(),
                    var.step()
                );
                self.indent += 1;
                self.print(var.body());
                self.indent -= 1;
                self.do_indent();
                println!("}}");
            }
            Expr::While(w) => {
                self.do_indent();
                println!("while {} {{", w.cond());
                self.indent += 1;
                self.print(w.body());
                self.indent -= 1;
                self.do_indent();
                println!("}}");
            }
            Expr::If(if_) => {
                self.do_indent();
                println!("if {} {{", if_.cond());
                self.indent += 1;
                self.print(if_.then());
                self.indent -= 1;
                self.do_indent();
                println!("}} else {{");
                self.indent += 1;
                self.print(if_.else_());
                self.indent -= 1;
                self.do_indent();
                println!("}}");
            }
            Expr::Add(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Sub(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Mul(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Div(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Mod(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Eq(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Ne(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Lt(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Le(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Gt(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Ge(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::And(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Or(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Not(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Xor(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Call(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Select(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Alloc(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Function(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Tensor(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Value(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::None => {}
            Expr::Str(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Variable(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Cast(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Min(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Max(a) => {
                self.do_indent();
                println!("{}", a);
            }
        }
    }
}
