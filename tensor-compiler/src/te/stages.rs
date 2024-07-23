use std::collections::HashMap;

use crate::{
    halide::{
        let_stmt::LetStmt,
        loop_utils::build_nested::build_nested_for,
        prime_expr::PrimeExpr,
        seq_stmt::Seq,
        stmt::Stmt,
        variable::Variable,
    },
    iter_var::IterVar,
};

#[derive(Clone)]
pub enum Body {
    PrimeExpr(PrimeExpr),
    Stmt(Stmt),
    Stage(usize),
}

pub struct Stage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) id: usize,
}

impl Stage {
    pub fn to_halide(&self, map: &HashMap<usize, Stage>) -> Stmt {
        let bodys = self.bodys.clone();
        let mut seq = Vec::new();
        for i in bodys {
            match i {
                crate::te::stages::Body::PrimeExpr(expr) => {
                    seq.push(
                        Stmt::LetStmt(
                            LetStmt::make(&Variable::make("placeholder"), expr, Stmt::None)
                        )
                    );
                }
                crate::te::stages::Body::Stmt(stmt) => {
                    seq.push(stmt);
                }
                crate::te::stages::Body::Stage(stage) => {
                    let stage = map.get(&stage).unwrap();
                    seq.push(stage.to_halide(&map));
                }
            }
        }
        build_nested_for(&self.dims, Stmt::Seq(Seq::make(seq)))
    }
}
