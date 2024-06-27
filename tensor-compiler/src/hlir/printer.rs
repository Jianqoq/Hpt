#![allow(dead_code)]
use super::expr::Expr;

pub struct HlirPrinter;

impl HlirPrinter {
    pub fn print<T: Into<Expr>>(&mut self, expr: T) {
        _HlirPrinter::new().print(expr, true);
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

    pub fn print<T: Into<Expr>>(&mut self, stmt: T, next_line: bool) {
        let stmt = stmt.into();
        match stmt {
            Expr::Let(let_) => {
                self.do_indent();
                let var = let_.var();
                let value = let_.value();
                match value {
                    Expr::Let(_) => {
                        panic!("let statement cannot have let statement as value");
                    }
                    Expr::For(_) | Expr::While(_) | Expr::If(_) => {
                        println!("let {} =", var);
                        self.print(value, false);
                    }
                    Expr::None => {
                        panic!("let statement cannot have None as value");
                    }
                    _ => {
                        print!("let {} = {}", var, value);
                    }
                }
                let body = let_.body();
                match body {
                    Expr::Let(_) | Expr::For(_) | Expr::While(_) | Expr::If(_) => {
                        println!(";");
                        self.print(body, true);
                    }
                    Expr::None => {
                        println!(";");
                    }
                    _ => {
                        print!(" in ");
                        self.print(body, true);
                    }
                }
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
                self.print(var.body(), true);
                self.indent -= 1;
                self.do_indent();
                if next_line {
                    println!("}}");
                } else {
                    print!("}}");
                }
            }
            Expr::While(w) => {
                self.do_indent();
                println!("while {} {{", w.cond());
                self.indent += 1;
                self.print(w.body(), true);
                self.indent -= 1;
                self.do_indent();
                if next_line {
                    println!("}}");
                } else {
                    print!("}}");
                }
            }
            Expr::If(if_) => {
                self.do_indent();
                println!("if {} {{", if_.cond());
                self.indent += 1;
                self.print(if_.then(), true);
                self.indent -= 1;
                self.do_indent();
                println!("}} else {{");
                self.indent += 1;
                self.print(if_.else_(), true);
                self.indent -= 1;
                self.do_indent();
                if next_line {
                    println!("}}");
                } else {
                    print!("}}");
                }
            }
            Expr::OpNode(a) => {
                self.do_indent();
                println!("{}", a);
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
                println!(
                    "fn {}({}) -> {} {{",
                    a.name(),
                    a
                        .args()
                        .iter()
                        .map(|x| format!("{}", x))
                        .collect::<Vec<String>>()
                        .join(", "),
                    a.return_type()
                );
                self.indent += 1;
                self.print(a.body(), true);
                self.indent -= 1;
                self.do_indent();
                if next_line {
                    println!("}}");
                } else {
                    print!("}}");
                }
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
            Expr::Tuple(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Type(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::TensorType(a) => {
                self.do_indent();
                println!("{}", a);
            }
            Expr::Slice(a) => {
                self.do_indent();
                println!("{}", a);
            }
        }
    }
}
