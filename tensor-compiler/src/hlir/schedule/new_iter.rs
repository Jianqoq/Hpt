#![allow(dead_code)]
#![allow(unused_imports)]

use std::{ cell::RefCell, collections::VecDeque, ops::Index, rc::Rc, sync::Arc };

use hashbrown::HashMap;

use crate::{
    halide::{
        let_stmt::LetStmt,
        loop_utils::build_nested::{ build_nested_for2, build_nested_for3 },
        prime_expr::PrimeExpr,
        printer::IRPrinter,
        seq_stmt::Seq,
        stmt::Stmt,
        variable::Variable,
    },
    hlir::tensor::Tensor,
};

pub type RcMut<T> = Rc<RefCell<T>>;

#[derive(Clone, Debug)]
pub enum Node {
    Base(BaseNode),
    Fused(FusedNode),
}

impl Node {
    pub fn push_child(&mut self, child: Rc<RefCell<Node>>) {
        match self {
            Node::Base(var) => {
                Arc::make_mut(&mut var.childs).push(child);
            }
            Node::Fused(fuse_var) => {
                Arc::make_mut(&mut fuse_var.childs).push(child);
            }
        }
    }
    pub fn var(&self) -> &Variable {
        match self {
            Node::Base(base) => &base.var,
            Node::Fused(fused) => &fused.var,
        }
    }
    pub fn end(&self) -> &PrimeExpr {
        match self {
            Node::Base(base) => &base.end,
            Node::Fused(fused) => &fused.end,
        }
    }
    pub fn start(&self) -> &PrimeExpr {
        match self {
            Node::Base(base) => &base.start,
            Node::Fused(fused) => &fused.start,
        }
    }
    pub fn set_ref_node(&mut self, node: RcMut<Node>) {
        match self {
            Node::Base(base) => {
                assert!(base.ref_node.is_none());
                base.ref_node = Some(node);
            }
            Node::Fused(fused) => {
                assert!(fused.ref_node.is_none());
                fused.ref_node = Some(node);
            }
        }
    }
}
#[derive(Clone, Debug)]
pub struct BaseNode {
    pub(crate) var: Variable,
    pub(crate) start: PrimeExpr,
    pub(crate) end: PrimeExpr,
    pub(crate) step: PrimeExpr,
    pub(crate) stride: PrimeExpr,
    pub(crate) parent: Option<RcMut<Node>>,
    pub(crate) childs: Arc<Vec<RcMut<Node>>>,
    pub(crate) expr: PrimeExpr,
    pub(crate) ref_node: Option<RcMut<Node>>,
}

impl BaseNode {
    pub fn make<A: Into<Variable>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<PrimeExpr>>(
        name: A,
        start: B,
        end: C,
        step: D,
        parent: Option<Rc<RefCell<Node>>>
    ) -> Self {
        BaseNode {
            var: name.into(),
            start: start.into(),
            end: end.into(),
            step: step.into(),
            stride: (1).into(),
            childs: vec![].into(),
            parent,
            expr: PrimeExpr::None,
            ref_node: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusedNode {
    pub(crate) lhs: RcMut<Node>,
    pub(crate) rhs: RcMut<Node>,
    pub(crate) var: Variable,
    pub(crate) start: PrimeExpr,
    pub(crate) end: PrimeExpr,
    pub(crate) step: PrimeExpr,
    pub(crate) childs: Arc<Vec<RcMut<Node>>>,
    pub(crate) expr: PrimeExpr,
    pub(crate) ref_node: Option<RcMut<Node>>,
}

impl FusedNode {
    pub fn make<A: Into<Variable>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<PrimeExpr>>(
        lhs: RcMut<Node>,
        rhs: RcMut<Node>,
        var: A,
        start: B,
        end: C,
        step: D
    ) -> Self {
        FusedNode {
            lhs,
            rhs,
            var: var.into(),
            start: start.into(),
            end: end.into(),
            step: step.into(),
            childs: vec![].into(),
            expr: PrimeExpr::None,
            ref_node: None,
        }
    }
}

#[derive(Clone)]
pub struct Stage {
    pub(crate) freezed_leaf: RcMut<Vec<RcMut<Node>>>, // these are from the parent stage, we can't do split/fuse on them, they are only used to generate indices
    pub(crate) leaf_id: RcMut<HashMap<usize, usize>>,
    pub(crate) id_leaf: RcMut<HashMap<usize, usize>>,
    pub(crate) address_map: RcMut<HashMap<usize, RcMut<Node>>>,
    pub(crate) attached_stage: RcMut<HashMap<usize, Vec<RcMut<Stage>>>>,
    pub(crate) transforms: VecDeque<Transforms>,
    pub(crate) body: PrimeExpr,
    pub(crate) name: Arc<String>,
}

impl Stage {
    pub fn split(&self, axis: &RcMut<Node>, factor: PrimeExpr) -> (RcMut<Node>, RcMut<Node>) {
        if
            self.leaf_id
                .borrow()
                .get(&(axis.as_ptr() as usize))
                .is_none()
        {
            panic!("axis is not in this stage");
        }
        let (outer, inner) = {
            match &*axis.borrow() {
                Node::Base(base) => {
                    let end = (&base.end + (&factor - 1)).floor_div(&factor);
                    let outer = Node::Base(
                        BaseNode::make(
                            format!("{}_outer", base.var),
                            &base.start,
                            end,
                            1,
                            Some(axis.clone())
                        )
                    );
                    let inner = Node::Base(
                        BaseNode::make(
                            format!("{}_inner", base.var),
                            0,
                            factor,
                            1,
                            Some(axis.clone())
                        )
                    );
                    (outer, inner)
                }
                Node::Fused(fused) => {
                    let end = (&fused.end + (&factor - 1)).floor_div(&factor);
                    let outer = Node::Base(
                        BaseNode::make(
                            format!("{}_outer", fused.var),
                            &fused.start,
                            end,
                            1,
                            Some(axis.clone())
                        )
                    );
                    let inner = Node::Base(
                        BaseNode::make(
                            format!("{}_inner", fused.var),
                            0,
                            factor,
                            1,
                            Some(axis.clone())
                        )
                    );
                    (outer, inner)
                }
            }
        };
        let outer = Rc::new(RefCell::new(outer));
        let inner = Rc::new(RefCell::new(inner));

        axis.borrow_mut().push_child(outer.clone());
        axis.borrow_mut().push_child(inner.clone());

        let axis_id = self.leaf_id.borrow()[&(axis.as_ptr() as usize)];
        for i in self.leaf_id.borrow_mut().values_mut() {
            if *i > axis_id {
                *i += 1;
            }
        }
        self.leaf_id.borrow_mut().remove(&(axis.as_ptr() as usize));
        self.leaf_id.borrow_mut().insert(outer.as_ptr() as usize, axis_id);
        self.leaf_id.borrow_mut().insert(inner.as_ptr() as usize, axis_id + 1);
        for (node_ptr, id) in self.leaf_id.borrow().iter() {
            self.id_leaf.borrow_mut().insert(*id, *node_ptr);
        }
        if
            self.attached_stage
                .borrow()
                .get(&(axis.as_ptr() as usize))
                .is_some()
        {
            let removed = self.attached_stage
                .borrow_mut()
                .remove(&(axis.as_ptr() as usize))
                .unwrap();
            self.attached_stage.borrow_mut().insert(inner.as_ptr() as usize, removed);
        }
        self.address_map.borrow_mut().insert(outer.as_ptr() as usize, outer.clone());
        self.address_map.borrow_mut().insert(inner.as_ptr() as usize, inner.clone());

        (outer, inner)
    }

    pub fn fuse(&self, axis1: &RcMut<Node>, axis2: &RcMut<Node>) -> RcMut<Node> {
        let axis1_id = self.leaf_id.borrow()[&(axis1.as_ptr() as usize)];
        let axis2_id = self.leaf_id.borrow()[&(axis2.as_ptr() as usize)];
        if axis1_id + 1 != axis2_id {
            panic!("axis1 and axis2 are not consecutive, axis1: {}, axis2: {}", axis1_id, axis2_id);
        }
        // check if two axis are in the same zone
        if
            self.attached_stage
                .borrow()
                .get(&(axis1.as_ptr() as usize))
                .is_some()
        {
            panic!("axis1 is attached to another stage");
        }
        let fused = Node::Fused(FusedNode {
            lhs: axis1.clone(),
            rhs: axis2.clone(),
            var: Variable::new(format!("fused_{}_{}", axis1.borrow().var(), axis2.borrow().var())),
            start: (0).into(),
            end: axis2.borrow().end() * axis1.borrow().end(),
            step: (1).into(),
            childs: vec![].into(),
            expr: PrimeExpr::None,
            ref_node: None,
        });
        let fused = Rc::new(RefCell::new(fused));
        axis1.borrow_mut().push_child(fused.clone());
        axis2.borrow_mut().push_child(fused.clone());
        self.leaf_id.borrow_mut().remove(&(axis1.as_ptr() as usize));
        self.leaf_id.borrow_mut().remove(&(axis2.as_ptr() as usize));
        self.leaf_id.borrow_mut().insert(fused.as_ptr() as usize, axis1_id);
        for id in self.leaf_id.borrow_mut().values_mut() {
            if *id > axis2_id {
                *id -= 1;
            }
        }
        for (node_ptr, id) in self.leaf_id.borrow().iter() {
            self.id_leaf.borrow_mut().insert(*id, *node_ptr);
        }
        self.address_map.borrow_mut().insert(fused.as_ptr() as usize, fused.clone());
        if
            self.attached_stage
                .borrow()
                .get(&(axis2.as_ptr() as usize))
                .is_some()
        {
            let removed = self.attached_stage
                .borrow_mut()
                .remove(&(axis2.as_ptr() as usize))
                .unwrap();
            self.attached_stage.borrow_mut().insert(fused.as_ptr() as usize, removed);
        }
        fused
    }

    pub fn reorder(&self, axes: &[RcMut<Node>]) {
        // check if all axes are in the same zone
        let ids = axes
            .iter()
            .map(|axis| self.leaf_id.borrow()[&(axis.as_ptr() as usize)])
            .collect::<Vec<_>>();
    }

    /// to inline axes must come from the stage itself
    pub fn compute_inline(
        &self,
        stage: &Stage,
        axes: &[RcMut<Node>],
        to_inline: &[RcMut<Node>]
    ) -> Option<RcMut<Stage>> {
        assert!(stage.freezed_leaf.borrow().len() == 0);

        let stage = stage.clone();

        let new_leaf_id = stage.leaf_id
            .borrow()
            .iter()
            .filter(|(node_ptr, _)| {
                if let Some(node) = stage.address_map.borrow().get(*node_ptr) {
                    !axes.iter().any(|x| x.as_ptr() == node.as_ptr())
                } else {
                    true
                }
            })
            .enumerate()
            .map(|(idx, (leaf, _))| (*leaf, idx))
            .collect::<HashMap<_, _>>();

        let new_id_leaf = new_leaf_id
            .iter()
            .map(|(node_ptr, id)| { (*id, *node_ptr) })
            .collect::<HashMap<_, _>>();

        *stage.leaf_id.borrow_mut() = new_leaf_id;
        *stage.id_leaf.borrow_mut() = new_id_leaf;

        to_inline
            .iter()
            .zip(axes.iter())
            .for_each(|(x, y)| { x.borrow_mut().set_ref_node(y.clone()) });

        stage.freezed_leaf.borrow_mut().extend(to_inline.iter().cloned());

        let ret = Rc::new(RefCell::new(stage));
        self.attached_stage
            .borrow_mut()
            .entry(axes.last().unwrap().as_ptr() as usize)
            .or_insert(Vec::new())
            .push(ret.clone());
        Some(ret)
    }

    pub fn tile(&self) -> (RcMut<Node>, RcMut<Node>) {
        todo!()
    }

    pub fn compute_root(&self) -> RcMut<Stage> {
        todo!()
    }

    pub fn axes(&self) -> Vec<RcMut<Node>> {
        self.leaf_id
            .borrow()
            .iter()
            .map(|(node_ptr, _)| { self.address_map.borrow()[&*node_ptr].clone() })
            .collect()
    }
    pub fn axis(&self, id: usize) -> RcMut<Node> {
        self.address_map.borrow()[&self.id_leaf.borrow()[&id].clone()].clone()
    }
}

fn all_elements_in_same_range(arr: &[usize], separators: &[usize]) -> bool {
    if arr.is_empty() {
        return true;
    }

    let range = find_range(arr[0], separators);
    for &elem in arr.iter().skip(1) {
        if find_range(elem, separators) != range {
            return false;
        }
    }
    true
}

fn find_range(value: usize, separators: &[usize]) -> Option<usize> {
    separators
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &sep)| value >= sep)
        .map(|(index, _)| index)
}

impl From<&Tensor> for Stage {
    fn from(value: &Tensor) -> Self {
        let root = value
            .shape()
            .iter()
            .map(|x| {
                let node = Node::Base(
                    BaseNode::make(x.var().clone(), x.start(), x.end(), x.step(), None)
                );
                Rc::new(RefCell::new(node))
            })
            .collect::<Vec<RcMut<Node>>>();
        let leaf_id = root
            .iter()
            .enumerate()
            .map(|(idx, x)| { (x.as_ptr() as usize, idx) })
            .collect::<HashMap<_, _>>();
        let id_leaf = leaf_id
            .iter()
            .map(|(x, y)| { (*y, *x) })
            .collect::<HashMap<_, _>>();
        let address_map = root
            .iter()
            .map(|x| { (x.as_ptr() as usize, x.clone()) })
            .collect::<HashMap<_, _>>();
        Stage {
            freezed_leaf: Rc::new(RefCell::new(vec![])),
            leaf_id: Rc::new(RefCell::new(leaf_id)),
            id_leaf: Rc::new(RefCell::new(id_leaf)),
            transforms: VecDeque::new(),
            name: Arc::new(value.name().to_string()),
            address_map: Rc::new(RefCell::new(address_map)),
            attached_stage: Rc::new(RefCell::new(HashMap::new())),
            body: value.body().clone(),
        }
    }
}

pub struct Schedule {
    pub stages: HashMap<Tensor, Stage>,
    pub records: VecDeque<Tensor>,
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            stages: HashMap::new(),
            records: VecDeque::new(),
        }
    }
    pub fn create(tensors: &[&Tensor]) -> Self {
        let mut schedule = Self::new();
        for tensor in tensors {
            let stage = Stage::from(*tensor);
            schedule.stages.insert((*tensor).clone(), stage);
        }
        schedule
    }
    pub fn split(
        &mut self,
        tensor: &Tensor,
        axis: &RcMut<Node>,
        inner_loop_size: impl Into<PrimeExpr>
    ) -> (RcMut<Node>, RcMut<Node>) {
        let inner_loop_size = inner_loop_size.into();
        let stages = self.stages.get_mut(tensor);
        if let Some(stages) = stages {
            stages.split(axis, inner_loop_size)
        } else {
            panic!("Schedule::split: tensor does not exist in temps_map");
        }
    }
    pub fn fuse(
        &mut self,
        tensor: &Tensor,
        axis1: &RcMut<Node>,
        axis2: &RcMut<Node>
    ) -> RcMut<Node> {
        let stages = self.stages.get_mut(tensor);
        if let Some(stages) = stages {
            stages.fuse(axis1, axis2)
        } else {
            panic!("Schedule::fuse: tensor does not exist in temps_map");
        }
    }
    pub fn compute_inline(
        &mut self,
        tensor: &Tensor,
        stage: &Stage,
        axes: &[RcMut<Node>],
        to_inline: &[RcMut<Node>]
    ) -> Option<RcMut<Stage>> {
        let stages = self.stages.get_mut(tensor);
        if let Some(stages) = stages {
            stages.compute_inline(stage, axes, to_inline)
        } else {
            panic!("Schedule::compute_inline: tensor does not exist in temps_map");
        }
    }
    pub fn reorder(&mut self, tensor: &Tensor, axes: &[&RcMut<Node>]) {
        let temp = self.stages.get_mut(tensor);
        if let Some(temp) = temp {
            let axes = axes
                .iter()
                .map(|x| (*x).clone())
                .collect::<Vec<_>>();
            temp.reorder(&axes);
        } else {
            panic!("Schedule::reorder: tensor does not exist in temps_map");
        }
    }

    pub fn execute_transforms(&mut self) {
        while let Some(t) = self.records.pop_front() {
            let stage = self.stages.get_mut(&t);
            if let Some(stage) = stage {
                let transform = stage.transforms
                    .pop_front()
                    .expect("Schedule::lower: transform is empty");
                match transform {
                    Transforms::Split(axis, inner_loop_size) => {
                        stage.split(&axis, inner_loop_size);
                    }
                    Transforms::Fuse(axis1, axis2) => {
                        stage.fuse(&axis1, &axis2);
                    }
                    Transforms::Reorder(axes) => {
                        stage.reorder(&axes);
                    }
                    Transforms::Inline(_) => todo!(),
                    Transforms::ComputeAt(_, _) => todo!(),
                    Transforms::Tile(_) => todo!(),
                }
            } else {
                panic!("Schedule::execute_transforms: tensor does not exist in temps_map");
            }
        }
    }

    pub fn to_halide(&self, tensor: &Tensor) -> Stmt {
        let body = self[tensor].body.clone();
        let let_stmt = LetStmt::make(&Variable::make(tensor.name()), body);
        build_nested_for3(Rc::new(RefCell::new(self[tensor].clone())), let_stmt.into())
    }
}

impl Index<&Tensor> for Schedule {
    type Output = Stage;

    fn index(&self, index: &Tensor) -> &Self::Output {
        self.stages.get(index).expect("Schedule::index: tensor does not exist in temps_map")
    }
}

#[derive(Clone)]
pub enum Transforms {
    Split(RcMut<Node>, PrimeExpr),
    Fuse(RcMut<Node>, RcMut<Node>),
    Reorder(Vec<RcMut<Node>>),
    Inline(Arc<String>),
    ComputeAt(Arc<String>, PrimeExpr),
    Tile(Vec<PrimeExpr>),
}

#[cfg(test)]
mod tests {
    use tensor_types::dtype::Dtype;

    use crate::hlir::tensor::compute;

    use super::*;

    #[test]
    fn test_schedule_inline() {
        let m = Variable::make("m");
        let n = Variable::make("n");
        let p = Variable::make("p");

        let a = Tensor::placeholder(&[&m, &n], Dtype::I64, "A");

        let c = compute(Dtype::BF16, [&n, &m], [&a], "C", |[a], [i, j]| {
            a.slice([&i, &j]) + a.slice([&j, &i])
        });
        let d = compute(Dtype::BF16, [&m, &p], [&c], "D", |[c], [i, j]| {
            c.slice([&i, &j]) + c.slice([&j, &i])
        });

        let mut s = Schedule::create(&[&a, &c, &d]);
        let (outer, inner) = s.split(&c, &s[&c].axis(0), 16);
        s.reorder(&c, &[&inner, &outer, &s[&c].axis(2)]);
        let (outer, inner) = s.split(&c, &s[&c].axis(0), 7);
        s.fuse(&c, &outer, &inner);
        IRPrinter.print_stmt(s.to_halide(&c));
    }
}
