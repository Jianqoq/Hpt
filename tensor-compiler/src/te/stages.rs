use std::collections::HashMap;

use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::Load,
        if_stmt::IfThenElse,
        loop_utils::build_nested::build_nested_for,
        prime_expr::PrimeExpr,
        seq_stmt::Seq,
        stmt::Stmt,
        traits::{ AccepterMutate, IRMutateVisitor, MutatorGetSet },
        variable::Variable,
    },
    iter_var::IterVar,
    te::subs_tensorload::SubsTensorLoadDims,
};

use super::insert_axes::InsertAxes;

#[derive(Clone)]
pub enum Body {
    Stmt(Stmt),
    Stage(Stage),
    ReduceStage(ReduceStage),
    If(If),
}

impl Body {
    pub fn id(&self) -> usize {
        match self {
            Body::Stmt(_) => panic!("stmt has no id"),
            Body::Stage(stage) => stage.id,
            Body::ReduceStage(reduce_stage) => reduce_stage.id,
            Body::If(if_) => if_.id,
        }
    }

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
    pub fn accept_mutate<V: IRMutateVisitor>(&mut self, subs_expr: &mut V) {
        match self {
            Body::Stmt(stmt) => {
                stmt.accept_mutate(subs_expr);
                *self = Body::Stmt(subs_expr.stmt().clone());
                subs_expr.set_stmt(Stmt::None);
                subs_expr.set_expr(PrimeExpr::None);
            }
            Body::Stage(stage) => stage.accept_mutate(subs_expr),
            Body::ReduceStage(red_stage) => {
                red_stage.accept_mutate(subs_expr);
            }
            Body::If(if_) => {
                if_.accept_mutate(subs_expr);
            }
        }
    }
    pub fn insert_new_axes(&mut self, axes: &mut InsertAxes) {
        match self {
            Body::Stmt(stmt) => {
                stmt.accept_mutate(axes);
                *self = Body::Stmt(axes.stmt().clone());
                axes.set_stmt(Stmt::None);
                axes.set_expr(PrimeExpr::None);
            }
            Body::Stage(stage) => {
                stage.insert_new_axes(axes);
            }
            Body::ReduceStage(red_stage) => {
                red_stage.insert_new_axes(axes);
            }
            Body::If(if_) => {
                if_.insert_new_axes(axes);
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
                Body::Stmt(stmt) => {
                    seq.push(stmt.clone());
                }
                Body::Stage(stage) => {
                    seq.push(stage.to_halide(&map));
                }
                Body::ReduceStage(reduce_stage) => {
                    seq.push(reduce_stage.to_halide(&map));
                }
                Body::If(if_) => {
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

    pub fn accept_mutate<V: IRMutateVisitor>(&mut self, subs_expr: &mut V) {
        for body in &mut self.bodys {
            body.accept_mutate(subs_expr);
        }
        for body in &mut self.inits {
            body.accept_mutate(subs_expr);
        }
        for body in &mut self.posts {
            body.accept_mutate(subs_expr);
        }
    }

    pub fn insert_new_axes(&mut self, axes: &mut InsertAxes) {
        for body in &mut self.bodys {
            body.insert_new_axes(axes);
        }
        for body in &mut self.inits {
            body.insert_new_axes(axes);
        }
        for body in &mut self.posts {
            body.insert_new_axes(axes);
        }
    }
}

#[derive(Clone)]
pub struct Stage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) dtype: Dtype,
    pub(crate) id: usize,
    pub(crate) out_id: usize,
    pub(crate) begins: Vec<PrimeExpr>,
    pub(crate) steps: Vec<PrimeExpr>,
    pub(crate) axes: Vec<PrimeExpr>,
}

impl Stage {
    pub fn to_halide(&self, map: &HashMap<usize, (Body, bool)>) -> Stmt {
        let bodys = self.bodys.clone();
        let mut seq = Vec::new();
        for body in bodys {
            match body {
                Body::Stmt(stmt) => {
                    seq.push(stmt.clone());
                }
                Body::Stage(stage) => {
                    seq.push(stage.to_halide(&map));
                }
                Body::ReduceStage(reduce_stage) => {
                    seq.push(reduce_stage.to_halide(&map));
                }
                Body::If(if_) => {
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

    pub fn accept_mutate<V: IRMutateVisitor>(&mut self, subs_expr: &mut V) {
        for body in &mut self.bodys {
            body.accept_mutate(subs_expr);
        }
    }

    pub fn insert_new_axes(&mut self, axes: &mut InsertAxes) {
        for body in &mut self.bodys {
            body.insert_new_axes(axes);
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
                Body::Stmt(stmt) => {
                    true_seq.push(stmt.clone());
                }
                Body::Stage(stage) => {
                    true_seq.push(stage.to_halide(&map));
                }
                Body::ReduceStage(reduce_stage) => {
                    true_seq.push(reduce_stage.to_halide(&map));
                }
                Body::If(if_) => {
                    true_seq.push(if_.to_halide(&map));
                }
            }
        }
        let true_stmt = Stmt::Seq(Seq::make(true_seq));
        let mut false_seq = Vec::new();
        for body in false_bodys {
            match body {
                Body::Stmt(stmt) => {
                    false_seq.push(stmt.clone());
                }
                Body::Stage(stage) => {
                    false_seq.push(stage.to_halide(&map));
                }
                Body::ReduceStage(reduce_stage) => {
                    false_seq.push(reduce_stage.to_halide(&map));
                }
                Body::If(if_) => {
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

    pub fn accept_mutate<V: IRMutateVisitor>(&mut self, subs_expr: &mut V) {
        for body in &mut self.true_bodys {
            body.accept_mutate(subs_expr);
        }
        for body in &mut self.false_bodys {
            body.accept_mutate(subs_expr);
        }
    }

    pub fn insert_new_axes(&mut self, axes: &mut InsertAxes) {
        for body in &mut self.true_bodys {
            body.insert_new_axes(axes);
        }
        for body in &mut self.false_bodys {
            body.insert_new_axes(axes);
        }
    }
}
