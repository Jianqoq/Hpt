use std::collections::HashMap;

use crate::{
    halide::{
        exprs::Load, if_stmt::IfThenElse, loop_utils::build_nested::build_nested_for, prime_expr::PrimeExpr, seq_stmt::Seq, stmt::Stmt, traits::{ AccepterMutate, MutatorGetSet }, variable::Variable
    },
    iter_var::IterVar,
    te::subs_tensorload::SubsTensorLoadDims,
};

#[derive(Clone)]
pub enum Body {
    Stmt(Stmt),
    Stage(Stage),
    ReduceStage(ReduceStage),
    If(If),
}

impl Body {
    pub fn to_halide(&self, map: &HashMap<usize, (Body, bool)>) -> Stmt {
        match self {
            Body::Stmt(stmt) => stmt.clone(),
            Body::Stage(stage) => stage.to_halide(map),
            Body::ReduceStage(reduce_stage) => reduce_stage.to_halide(map),
            Body::If(if_) => if_.to_halide(map),
        }
    }
    pub fn all_parents_dims(&self, map: &HashMap<usize, (Body, bool)>) -> Vec<IterVar> {
        match self {
            Body::Stmt(_) => Vec::new(),
            Body::Stage(stage) => stage.dims.clone(),
            Body::ReduceStage(reduce_stage) => {
                let mut dims = reduce_stage.dims.clone();
                let input = reduce_stage.input;
                let (body, _) = map.get(&input).unwrap();
                let mut parent_dims = body.all_parents_dims(map);
                parent_dims.append(&mut dims);
                dims
            }
            Body::If(if_) => {
                let (body, _) = map.get(&if_.input).unwrap();
                body.all_parents_dims(map)
            }
        }
    }
}

#[derive(Clone)]
pub struct ReduceStage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) id: usize,
    pub(crate) inits: Vec<Body>,
    pub(crate) posts: Vec<Body>,
    pub(crate) input: usize,
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
                crate::te::stages::Body::If(if_) => {
                    seq.push(if_.to_halide(&map));
                }
            }
        }
        let for_stmt = build_nested_for(&self.dims, Stmt::Seq(Seq::make(seq)));
        let inits = self.inits
            .iter()
            .map(|init| init.to_halide(map))
            .collect::<Vec<Stmt>>();
        let posts = self.posts
            .iter()
            .map(|post| post.to_halide(map))
            .collect::<Vec<Stmt>>();
        Stmt::Seq(
            Seq::make(vec![Stmt::Seq(Seq::make(inits)), for_stmt, Stmt::Seq(Seq::make(posts))])
        )
    }
    pub fn broadcast_new_dims(&mut self, strides: &Vec<PrimeExpr>, axes: &Vec<PrimeExpr>) {
        let mut subs_tensorload = SubsTensorLoadDims::new(strides, axes);
        for body in &mut self.bodys {
            match body {
                Body::Stmt(stmt) => {
                    stmt.accept_mutate(&mut subs_tensorload);
                    *body = Body::Stmt(subs_tensorload.stmt().clone());
                }
                Body::Stage(stage) => stage.broadcast_new_dims(strides, axes),
                Body::ReduceStage(red_stage) => {
                    let red_strides = (strides.len()..strides.len() + red_stage.dims.len())
                        .map(|x| {
                            Load::make(Variable::new(format!("%{}.s", red_stage.id)), x).into()
                        })
                        .collect::<Vec<PrimeExpr>>();
                    let new_strides = strides
                        .iter()
                        .chain(red_strides.iter())
                        .cloned()
                        .collect::<Vec<PrimeExpr>>();
                    let new_axes = axes
                        .iter()
                        .cloned()
                        .chain(red_stage.dims.iter().map(|x| x.var().into()))
                        .collect::<Vec<PrimeExpr>>();
                    red_stage.broadcast_new_dims(&new_strides, &new_axes);
                }
                Body::If(if_) => {
                    if_.broadcast_new_dims(strides, axes);
                }
            }
        }
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
                crate::te::stages::Body::If(if_) => {
                    seq.push(if_.to_halide(&map));
                }
            }
        }
        build_nested_for(&self.dims, Stmt::Seq(Seq::make(seq)))
    }
    pub fn broadcast_new_dims(&mut self, strides: &Vec<PrimeExpr>, axes: &Vec<PrimeExpr>) {
        let mut subs_tensorload = SubsTensorLoadDims::new(strides, axes);
        for body in &mut self.bodys {
            match body {
                Body::Stmt(stmt) => {
                    stmt.accept_mutate(&mut subs_tensorload);
                    *body = Body::Stmt(subs_tensorload.stmt().clone());
                }
                Body::Stage(stage) => stage.broadcast_new_dims(strides, axes),
                Body::ReduceStage(red_stage) => {
                    let red_strides = (strides.len()..strides.len() + red_stage.dims.len())
                        .map(|x| {
                            Load::make(Variable::new(format!("%{}.s", red_stage.id)), x).into()
                        })
                        .collect::<Vec<PrimeExpr>>();
                    let new_strides = strides
                        .iter()
                        .chain(red_strides.iter())
                        .cloned()
                        .collect::<Vec<PrimeExpr>>();
                    let new_axes = axes
                        .iter()
                        .cloned()
                        .chain(red_stage.dims.iter().map(|x| x.var().into()))
                        .collect::<Vec<PrimeExpr>>();
                    red_stage.broadcast_new_dims(&new_strides, &new_axes);
                }
                Body::If(if_) => {
                    if_.broadcast_new_dims(strides, axes);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct If {
    pub(crate) cond: PrimeExpr,
    pub(crate) true_bodys: Vec<Body>,
    pub(crate) false_bodys: Vec<Body>,
    pub(crate) id: usize,
    pub(crate) input: usize,
}

impl If {
    pub fn to_halide(&self, map: &HashMap<usize, (Body, bool)>) -> Stmt {
        let true_bodys = self.true_bodys.clone();
        let false_bodys = self.false_bodys.clone();
        let mut true_seq = Vec::new();
        for body in true_bodys {
            match body {
                crate::te::stages::Body::Stmt(stmt) => {
                    true_seq.push(stmt.clone());
                }
                crate::te::stages::Body::Stage(stage) => {
                    true_seq.push(stage.to_halide(&map));
                }
                crate::te::stages::Body::ReduceStage(reduce_stage) => {
                    true_seq.push(reduce_stage.to_halide(&map));
                }
                crate::te::stages::Body::If(if_) => {
                    true_seq.push(if_.to_halide(&map));
                }
            }
        }
        let true_stmt = Stmt::Seq(Seq::make(true_seq));
        let mut false_seq = Vec::new();
        for body in false_bodys {
            match body {
                crate::te::stages::Body::Stmt(stmt) => {
                    false_seq.push(stmt.clone());
                }
                crate::te::stages::Body::Stage(stage) => {
                    false_seq.push(stage.to_halide(&map));
                }
                crate::te::stages::Body::ReduceStage(reduce_stage) => {
                    false_seq.push(reduce_stage.to_halide(&map));
                }
                crate::te::stages::Body::If(if_) => {
                    false_seq.push(if_.to_halide(&map));
                }
            }
        }
        let false_stmt = Stmt::Seq(Seq::make(false_seq));
        Stmt::IfThenElse(IfThenElse::make(self.cond.clone(), true_stmt, false_stmt))
    }

    pub fn broadcast_new_dims(&mut self, strides: &Vec<PrimeExpr>, axes: &Vec<PrimeExpr>) {
        let mut subs_tensorload = SubsTensorLoadDims::new(strides, axes);
        for body in &mut self.true_bodys {
            match body {
                Body::Stmt(stmt) => {
                    stmt.accept_mutate(&mut subs_tensorload);
                    *body = Body::Stmt(subs_tensorload.stmt().clone());
                }
                Body::Stage(stage) => stage.broadcast_new_dims(strides, axes),
                Body::ReduceStage(red_stage) => {
                    let red_strides = (strides.len()..strides.len() + red_stage.dims.len())
                        .map(|x| {
                            Load::make(Variable::new(format!("%{}.s", red_stage.id)), x).into()
                        })
                        .collect::<Vec<PrimeExpr>>();
                    let new_strides = strides
                        .iter()
                        .chain(red_strides.iter())
                        .cloned()
                        .collect::<Vec<PrimeExpr>>();
                    let new_axes = axes
                        .iter()
                        .cloned()
                        .chain(red_stage.dims.iter().map(|x| x.var().into()))
                        .collect::<Vec<PrimeExpr>>();
                    red_stage.broadcast_new_dims(&new_strides, &new_axes);
                }
                Body::If(if_) => {
                    for body in &mut if_.true_bodys {
                        match body {
                            Body::Stmt(stmt) => {
                                stmt.accept_mutate(&mut subs_tensorload);
                                *body = Body::Stmt(subs_tensorload.stmt().clone());
                            }
                            Body::Stage(stage) => stage.broadcast_new_dims(strides, axes),
                            Body::ReduceStage(red_stage) => {
                                let red_strides = (strides.len()..strides.len() +
                                    red_stage.dims.len())
                                    .map(|x| {
                                        Load::make(
                                            Variable::new(format!("%{}.s", red_stage.id)),
                                            x
                                        ).into()
                                    })
                                    .collect::<Vec<PrimeExpr>>();
                                let new_strides = strides
                                    .iter()
                                    .chain(red_strides.iter())
                                    .cloned()
                                    .collect::<Vec<PrimeExpr>>();
                                let new_axes = axes
                                    .iter()
                                    .cloned()
                                    .chain(red_stage.dims.iter().map(|x| x.var().into()))
                                    .collect::<Vec<PrimeExpr>>();
                                red_stage.broadcast_new_dims(&new_strides, &new_axes);
                            }
                            Body::If(if_) => {
                                if_.broadcast_new_dims(strides, axes);
                            }
                        }
                    }
                    for body in &mut if_.false_bodys {
                        match body {
                            Body::Stmt(stmt) => {
                                stmt.accept_mutate(&mut subs_tensorload);
                                *body = Body::Stmt(subs_tensorload.stmt().clone());
                            }
                            Body::Stage(stage) => stage.broadcast_new_dims(strides, axes),
                            Body::ReduceStage(red_stage) => {
                                let red_strides = (strides.len()..strides.len() +
                                    red_stage.dims.len())
                                    .map(|x| {
                                        Load::make(
                                            Variable::new(format!("%{}.s", red_stage.id)),
                                            x
                                        ).into()
                                    })
                                    .collect::<Vec<PrimeExpr>>();
                                let new_strides = strides
                                    .iter()
                                    .chain(red_strides.iter())
                                    .cloned()
                                    .collect::<Vec<PrimeExpr>>();
                                let new_axes = axes
                                    .iter()
                                    .cloned()
                                    .chain(red_stage.dims.iter().map(|x| x.var().into()))
                                    .collect::<Vec<PrimeExpr>>();
                                red_stage.broadcast_new_dims(&new_strides, &new_axes);
                            }
                            Body::If(if_) => {
                                if_.broadcast_new_dims(strides, axes);
                            }
                        }
                    }
                }
            }
        }
    }
}
