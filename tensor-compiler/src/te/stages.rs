use std::collections::HashMap;

use crate::{
    halide::{
        let_stmt::LetStmt,
        loop_utils::build_nested::build_nested_for,
        seq_stmt::Seq,
        stmt::Stmt,
        variable::Variable,
    },
    iter_var::IterVar,
};

#[derive(Clone)]
pub enum Body {
    Stmt(Stmt),
    Stage(Stage),
    ReduceStage(ReduceStage),
}

impl Body {
    pub fn to_halide(&self, map: &HashMap<usize, (Body, bool)>) -> Stmt {
        match self {
            Body::Stmt(stmt) => stmt.clone(),
            Body::Stage(stage) => stage.to_halide(map),
            Body::ReduceStage(reduce_stage) => reduce_stage.to_halide(map),
        }
    }
}

#[derive(Clone)]
pub struct ReduceStage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) id: usize,
    pub(crate) inits: Vec<Body>,
}

impl ReduceStage {
    pub fn to_halide(&self, map: &HashMap<usize, (Body, bool)>) -> Stmt {
        let bodys = self.bodys.clone();
        let mut seq = Vec::new();
        for body in bodys {
            match body {
                crate::te::stages::Body::Stmt(stmt) => {
                    seq.push(stmt.clone());
                }
                crate::te::stages::Body::Stage(stage) => {
                    seq.push(stage.to_halide(&map));
                }
                crate::te::stages::Body::ReduceStage(reduce_stage) => {
                    seq.push(reduce_stage.to_halide(&map));
                }
            }
        }
        todo!()
    }
}

#[derive(Clone)]
pub struct Stage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) id: usize,
}

impl Stage {
    pub fn to_halide(&self, map: &HashMap<usize, (Body, bool)>) -> Stmt {
        let bodys = self.bodys.clone();
        let mut seq = Vec::new();
        for body in bodys {
            match body {
                crate::te::stages::Body::Stmt(stmt) => {
                    seq.push(stmt.clone());
                }
                crate::te::stages::Body::Stage(stage) => {
                    seq.push(stage.to_halide(&map));
                }
                crate::te::stages::Body::ReduceStage(reduce_stage) => {
                    seq.push(reduce_stage.to_halide(&map));
                }
            }
        }
        build_nested_for(&self.dims, Stmt::Seq(Seq::make(seq)))
    }
}
