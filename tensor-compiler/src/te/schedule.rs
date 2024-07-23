use std::{ collections::HashMap, fmt::{ self, Display, Formatter } };

use crate::halide::{ printer::IRPrinter, stmt::Stmt };

use super::stages::Body;

pub struct Schedule {
    pub(crate) qa: HashMap<usize, (Body, bool)>,
}

impl Schedule {
    pub fn to_halide(&self) -> Vec<Stmt> {
        let mut ret = Vec::new();
        for (body, is_output) in self.qa.values() {
            match body {
                Body::Stage(stage) => {
                    if *is_output {
                        ret.push(stage.to_halide(&self.qa));
                    }
                }
                _ => {}
            }
        }
        ret
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self
                .to_halide()
                .iter()
                .map(|x| IRPrinter.print_stmt_str(x))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}
