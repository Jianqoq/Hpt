use std::{ collections::HashMap, sync::Arc };

use crate::{
    halide::{
        exprs::Load,
        loop_utils::build_nested::build_nested_for,
        prime_expr::PrimeExpr,
        seq_stmt::Seq,
        stmt::Stmt,
        traits::{ AccepterMutate, MutatorGetSet },
        variable::Variable,
    },
    iter_var::IterVar,
    te::subs_tensorload::SubsTensorLoadDims,
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
    pub fn broadcast_new_dims(
        &mut self,
        begins: &Vec<PrimeExpr>,
        steps: &Vec<PrimeExpr>,
        strides: &Vec<PrimeExpr>,
        axes: &Vec<PrimeExpr>
    ) {
        let mut subs_tensorload = SubsTensorLoadDims::new(begins, steps, strides, axes);
        for body in &mut self.bodys {
            match body {
                Body::Stmt(stmt) => {
                    stmt.accept_mutate(&mut subs_tensorload);
                    *body = Body::Stmt(subs_tensorload.stmt().clone());
                }
                Body::Stage(stage) => stage.broadcast_new_dims(begins, steps, strides, axes),
                Body::ReduceStage(red_stage) => {
                    let red_beings = (0..red_stage.dims.len())
                        .map(|_| (0i64).into())
                        .collect::<Vec<PrimeExpr>>();
                    let red_steps = (0..red_stage.dims.len())
                        .map(|_| (1i64).into())
                        .collect::<Vec<PrimeExpr>>();
                    let red_strides = (strides.len()..strides.len() + red_stage.dims.len())
                        .map(|x| {
                            Load::make(Variable::new(format!("%{}.s", red_stage.id)), x).into()
                        })
                        .collect::<Vec<PrimeExpr>>();
                    let new_begins = begins
                        .iter()
                        .chain(red_beings.iter())
                        .cloned()
                        .collect::<Vec<PrimeExpr>>();
                    let new_steps = steps
                        .iter()
                        .chain(red_steps.iter())
                        .cloned()
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
                    red_stage.broadcast_new_dims(&new_begins, &new_steps, &new_strides, &new_axes);
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
            }
        }
        build_nested_for(&self.dims, Stmt::Seq(Seq::make(seq)))
    }
    pub fn broadcast_new_dims(
        &mut self,
        begins: &Vec<PrimeExpr>,
        steps: &Vec<PrimeExpr>,
        strides: &Vec<PrimeExpr>,
        axes: &Vec<PrimeExpr>
    ) {
        let mut subs_tensorload = SubsTensorLoadDims::new(begins, steps, strides, axes);
        for body in &mut self.bodys {
            match body {
                Body::Stmt(stmt) => {
                    stmt.accept_mutate(&mut subs_tensorload);
                    *body = Body::Stmt(subs_tensorload.stmt().clone());
                }
                Body::Stage(stage) => stage.broadcast_new_dims(begins, steps, strides, axes),
                Body::ReduceStage(red_stage) => {
                    let red_beings = (0..red_stage.dims.len())
                        .map(|_| (0i64).into())
                        .collect::<Vec<PrimeExpr>>();
                    let red_steps = (0..red_stage.dims.len())
                        .map(|_| (1i64).into())
                        .collect::<Vec<PrimeExpr>>();
                    let red_strides = (strides.len()..strides.len() + red_stage.dims.len())
                        .map(|x| {
                            Load::make(Variable::new(format!("%{}.s", red_stage.id)), x).into()
                        })
                        .collect::<Vec<PrimeExpr>>();
                    let new_begins = begins
                        .iter()
                        .chain(red_beings.iter())
                        .cloned()
                        .collect::<Vec<PrimeExpr>>();
                    let new_steps = steps
                        .iter()
                        .chain(red_steps.iter())
                        .cloned()
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
                    red_stage.broadcast_new_dims(&new_begins, &new_steps, &new_strides, &new_axes);
                }
            }
        }
    }
}
