#![allow(dead_code)]
#![allow(unused_imports)]

use std::{ cell::RefCell, collections::VecDeque, fmt::{ Debug, Display }, rc::Rc, sync::Arc };

use hashbrown::{ HashMap, HashSet };
use tensor_types::dtype::Dtype;

use crate::{
    edges::Edges,
    halide::{ exprs::{ Add, FloorDiv, Int, Mod, Mul }, prime_expr::PrimeExpr, variable::Variable },
    to_prim_expr::ToPrimeExpr,
};
#[derive(Clone)]
pub struct IterVar {
    pub(crate) var: Variable,
    pub(crate) start: PrimeExpr,
    pub(crate) end: PrimeExpr,
    pub(crate) step: PrimeExpr,
    pub(crate) stride: PrimeExpr,
    pub(crate) parent: Option<Rc<RefCell<Iter>>>,
    pub(crate) childs: Arc<Vec<Rc<RefCell<Iter>>>>,
}

impl IterVar {
    pub fn make<A: Into<Variable>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<PrimeExpr>>(
        name: A,
        start: B,
        end: C,
        step: D,
        parent: Option<Rc<RefCell<Iter>>>
    ) -> Self {
        IterVar {
            var: name.into(),
            start: start.into(),
            end: end.into(),
            step: step.into(),
            stride: (1).into(),
            childs: vec![].into(),
            parent,
        }
    }
}
#[derive(Clone)]
pub struct FuseVar {
    pub(crate) lhs: Rc<RefCell<Iter>>,
    pub(crate) rhs: Rc<RefCell<Iter>>,
    pub(crate) var: Variable,
    pub(crate) start: PrimeExpr,
    pub(crate) end: PrimeExpr,
    pub(crate) step: PrimeExpr,
    pub(crate) childs: Arc<Vec<Rc<RefCell<Iter>>>>,
}

#[derive(Clone)]
pub enum Iter {
    IterVar(IterVar),
    FuseVar(FuseVar),
}

impl Display for Iter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Iter::IterVar(var) => write!(f, "for {} in {}..{}", var.var.name(), var.start, var.end),
            Iter::FuseVar(fuse_var) =>
                write!(f, "for {} in {}..{}", fuse_var.var.name(), fuse_var.start, fuse_var.end),
        }
    }
}

impl Debug for Iter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Iter::IterVar(var) => write!(f, "for {} in {}..{}", var.var.name(), var.start, var.end),
            Iter::FuseVar(fuse_var) =>
                write!(f, "for {} in {}..{}", fuse_var.var.name(), fuse_var.start, fuse_var.end),
        }
    }
}

impl Iter {
    pub fn childs(&self) -> Vec<&Rc<RefCell<Iter>>> {
        match self {
            Iter::IterVar(var) => var.childs.iter().collect(),
            Iter::FuseVar(fuse_var) => fuse_var.childs.iter().collect(),
        }
    }
    pub fn childs_(&self) -> Vec<Rc<RefCell<Iter>>> {
        match self {
            Iter::IterVar(var) => var.childs.as_ref().clone(),
            Iter::FuseVar(fuse_var) => fuse_var.childs.as_ref().clone(),
        }
    }
    pub fn var(&self) -> &Variable {
        match self {
            Iter::IterVar(var) => &var.var,
            Iter::FuseVar(fuse_var) => &fuse_var.var,
        }
    }
    pub fn start(&self) -> &PrimeExpr {
        match self {
            Iter::IterVar(var) => &var.start,
            Iter::FuseVar(fuse_var) => &fuse_var.start,
        }
    }

    pub fn end(&self) -> &PrimeExpr {
        match self {
            Iter::IterVar(var) => &var.end,
            Iter::FuseVar(fuse_var) => &fuse_var.end,
        }
    }

    pub fn step(&self) -> &PrimeExpr {
        match self {
            Iter::IterVar(var) => &var.step,
            Iter::FuseVar(fuse_var) => &fuse_var.step,
        }
    }

    pub fn push_child(&mut self, child: Rc<RefCell<Iter>>) {
        match self {
            Iter::IterVar(var) => {
                Arc::make_mut(&mut var.childs).push(child);
            }
            Iter::FuseVar(fuse_var) => {
                Arc::make_mut(&mut fuse_var.childs).push(child);
            }
        }
    }
}

pub fn gen_edges(shape: &Vec<Rc<RefCell<Iter>>>) {
    let mut m = HashMap::new();
    let mut m_inv = HashMap::new();
    let mut cnt = 0usize;
    let mut visited = HashSet::new();
    let mut edges = Edges::new();
    for root in shape.iter() {
        let mut stack = vec![root.clone()];
        while let Some(node) = stack.pop() {
            m.insert(cnt, node.clone());
            m_inv.insert(node.as_ptr(), cnt);
            cnt += 1;
            for child in node.borrow().childs() {
                let ptr = child.as_ptr();
                if visited.contains(&ptr) {
                    continue;
                } else {
                    visited.insert(ptr);
                }
                stack.push(child.clone());
            }
        }
    }
    for (k, v) in m.iter() {
        let childs = v.borrow().childs_();
        edges
            .entry(*k)
            .or_insert(HashSet::new())
            .extend(childs.iter().map(|x| m_inv.get(&x.as_ptr()).unwrap().clone()));
    }
    println!("{:#?}", edges);
    println!("{:#?}", edges.invert());
    for (k, v) in m.iter() {
        println!("{}: {}", k, v.borrow().var());
    }
    let sorted = topo(&edges, &m);
    println!("{:?}", sorted);
    let mut expr_map = HashMap::<usize, PrimeExpr>::new();
    if let Some(sorted) = sorted {
        for i in sorted {
            let node = &m[&i];
            match &*node.borrow() {
                Iter::IterVar(iter_var) => {
                    if let Some(parent) = &iter_var.parent {
                        match &mut *parent.borrow_mut() {
                            Iter::IterVar(_parent) => {
                                // this is a splitting operation, parent must have 2 children
                                assert!(_parent.childs.len() == 2);
                                let parent_childs = &_parent.childs;
                                // check node is lhs of parent children or rhs of parent children
                                let is_rhs = parent_childs[1].as_ptr() == node.as_ptr();
                                let key = m_inv[&parent.as_ptr()];
                                if let Some(node_expr) = expr_map.get(&i) {
                                    let node_expr = node_expr.clone();
                                    if let Some(parent_expr) = expr_map.get_mut(&key) {
                                        assert!(parent_expr.is_add());
                                        // it has been visited
                                        if is_rhs {
                                            // rhs_expr is ready, and since the expr is add, the rhs must be None
                                            assert!(parent_expr.to_add().unwrap().e2().is_none());
                                            let mut to_add = parent_expr.to_add().unwrap().clone();
                                            to_add.set_e2(node_expr);
                                            *parent_expr = to_add.into();
                                        } else {
                                            // lhs_expr is ready, and since the expr is add, the lhs must be None
                                            assert!(parent_expr.to_add().unwrap().e1().is_none());
                                            let mut to_add = parent_expr.to_add().unwrap().clone();
                                            to_add.set_e1(node_expr);
                                            *parent_expr = to_add.into();
                                        }
                                    } else {
                                        if is_rhs {
                                            let add = Add::make(PrimeExpr::None, node_expr);
                                            expr_map.insert(key, add.into());
                                        } else {
                                            let add = Add::make(node_expr, PrimeExpr::None);
                                            expr_map.insert(key, add.into());
                                        }
                                    }
                                } else {
                                    // current node doesn't have accumulated expr, based on the topo order, it must be a leaf node
                                    assert!(node.borrow().childs().len() == 0);
                                    if let Some(expr) = expr_map.get_mut(&key) {
                                        assert!(expr.is_add());
                                        if is_rhs {
                                            assert!(expr.to_add().unwrap().e2().is_none());
                                            let mut to_add = expr.to_add().unwrap().clone();
                                            to_add.set_e2(node.borrow().var().clone());
                                            *expr = to_add.into();
                                        } else {
                                            assert!(expr.to_add().unwrap().e1().is_none());
                                            let mut to_add = expr.to_add().unwrap().clone();
                                            // we need to get the rhs end
                                            let rhs_end = parent_childs[1].borrow().end().clone();
                                            to_add.set_e1(Mul::make(node.borrow().var(), rhs_end));
                                            *expr = to_add.into();
                                        }
                                    } else {
                                        if is_rhs {
                                            let add = Add::make(
                                                PrimeExpr::None,
                                                node.borrow().var().clone()
                                            );
                                            expr_map.insert(key, add.into());
                                        } else {
                                            let rhs_end = parent_childs[1].borrow().end().clone();
                                            let add = Add::make(
                                                Mul::make(node.borrow().var(), rhs_end),
                                                PrimeExpr::None
                                            );
                                            expr_map.insert(key, add.into());
                                        }
                                    }
                                }
                            }
                            Iter::FuseVar(_parent) => {
                                // this is a splitting operation, parent must have 2 children
                                assert!(_parent.childs.len() == 2);
                                let parent_childs = &_parent.childs;
                                // check node is lhs of parent children or rhs of parent children
                                let is_rhs = parent_childs[1].as_ptr() == node.as_ptr();
                                let key = m_inv[&parent.as_ptr()];
                                if let Some(node_expr) = expr_map.get(&i) {
                                    let node_expr = node_expr.clone();
                                    if let Some(expr) = expr_map.get_mut(&key) {
                                        assert!(expr.is_add());
                                        // it has been visited
                                        if is_rhs {
                                            // rhs_expr is ready, and since the expr is add, the rhs must be None
                                            assert!(expr.to_add().unwrap().e2().is_none());
                                            let mut to_add = expr.to_add().unwrap().clone();
                                            to_add.set_e2(node_expr);
                                            *expr = to_add.into();
                                        } else {
                                            // lhs_expr is ready, and since the expr is add, the lhs must be None
                                            assert!(expr.to_add().unwrap().e1().is_none());
                                            let mut to_add = expr.to_add().unwrap().clone();
                                            to_add.set_e1(node_expr);
                                            *expr = to_add.into();
                                        }
                                    } else {
                                        if is_rhs {
                                            let add = Add::make(PrimeExpr::None, node_expr);
                                            expr_map.insert(key, add.into());
                                        } else {
                                            let add = Add::make(node_expr, PrimeExpr::None);
                                            expr_map.insert(key, add.into());
                                        }
                                    }
                                } else {
                                    // current node doesn't have accumulated expr, based on the topo order, it must be a leaf node
                                    assert!(node.borrow().childs().len() == 0);
                                    if let Some(expr) = expr_map.get_mut(&key) {
                                        assert!(expr.is_add());
                                        if is_rhs {
                                            assert!(expr.to_add().unwrap().e2().is_none());
                                            let mut to_add = expr.to_add().unwrap().clone();
                                            to_add.set_e2(node.borrow().var());
                                            *expr = to_add.into();
                                        } else {
                                            assert!(expr.to_add().unwrap().e1().is_none());
                                            let mut to_add = expr.to_add().unwrap().clone();
                                            // we need to get the rhs end
                                            let rhs_end = parent_childs[1].borrow().end().clone();
                                            to_add.set_e1(Mul::make(node.borrow().var(), rhs_end));
                                            *expr = to_add.into();
                                        }
                                    } else {
                                        if is_rhs {
                                            let add = Add::make(
                                                PrimeExpr::None,
                                                node.borrow().end().clone()
                                            );
                                            expr_map.insert(key, add.into());
                                        } else {
                                            let rhs_end = parent_childs[1].borrow().end().clone();
                                            let add = Add::make(
                                                Mul::make(node.borrow().var(), rhs_end),
                                                PrimeExpr::None
                                            );
                                            expr_map.insert(key, add.into());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Iter::FuseVar(fused) => {
                    let lhs = m_inv[&fused.lhs.as_ptr()];
                    let rhs = m_inv[&fused.rhs.as_ptr()];

                    // because we are using topo order, lhs and rhs must not be visited
                    assert!(expr_map.get_mut(&lhs).is_none());
                    assert!(expr_map.get_mut(&rhs).is_none());
                    let lhs_expr;
                    let rhs_expr;
                    if let Some(node_expr) = expr_map.get(&i) {
                        lhs_expr = FloorDiv::make(
                            &node_expr,
                            m[&rhs].borrow().end().clone()
                        ).into();
                        rhs_expr = node_expr % m[&rhs].borrow().end();
                    } else {
                        // current node doesn't have accumulated expr, based on the topo order, it must be a leaf node
                        assert!(node.borrow().childs().len() == 0);
                        lhs_expr = FloorDiv::make(
                            &fused.var,
                            m[&rhs].borrow().end().clone()
                        ).into();
                        rhs_expr = Mod::make(&fused.var, m[&rhs].borrow().end()).into();
                    }
                    expr_map.insert(rhs, rhs_expr);
                    expr_map.insert(lhs, lhs_expr);
                }
            }
        }
    } else {
        panic!("cycle detected");
    }
    for (k, v) in expr_map.iter() {
        println!("{}: {}", k, v);
    }
}

fn topo(
    edges: &Edges<usize>,
    nodes: &HashMap<usize, Rc<RefCell<Iter>>>
) -> Option<VecDeque<usize>> {
    let mut in_degree = HashMap::new();
    let mut queue = VecDeque::new();
    let mut order = VecDeque::new();
    let edges = edges.invert();
    // calculate in degree
    for (&node_id, _) in nodes.iter() {
        in_degree.entry(node_id).or_insert(0);
        let edges = edges.get(&node_id);
        if let Some(edges) = edges {
            for &target in edges {
                *in_degree.entry(target).or_insert(0) += 1;
            }
        }
    }

    // push nodes with in degree 0 to queue
    for (&node_id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node_id);
        }
    }

    // topological sort
    while let Some(node_id) = queue.pop_front() {
        order.push_back(node_id);
        if let Some(_) = nodes.get(&node_id) {
            let edges = edges.get(&node_id);
            if let Some(edges) = edges {
                for &target in edges {
                    let degree = in_degree.get_mut(&target).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(target);
                    }
                }
            }
        }
    }

    // check if there is a cycle
    if order.len() == nodes.len() {
        Some(order)
    } else {
        None // cycle detected
    }
}

pub fn collect_available_axes(
    shape: &Vec<Rc<RefCell<Iter>>>,
    visited: &mut HashSet<*mut Iter>
) -> Vec<Rc<RefCell<Iter>>> {
    let mut available_axes = vec![];
    for iter in shape.iter() {
        if iter.borrow().childs().is_empty() {
            let ptr = iter.as_ptr();
            if visited.contains(&ptr) {
                continue;
            } else {
                visited.insert(ptr);
            }
            available_axes.push(iter.clone());
        } else {
            let mut _available_axes = collect_available_axes(&iter.borrow().childs_(), visited);
            if !_available_axes.is_empty() {
                available_axes.extend(_available_axes);
            }
        }
    }
    available_axes
}

pub fn pack_groups(
    shape: &Vec<Rc<RefCell<Iter>>>,
    visited: &mut HashSet<*mut Iter>,
    groups: &mut Vec<Vec<usize>>,
    root: bool
) {
    for (idx, iter) in shape.iter().enumerate() {
        if iter.borrow().childs().is_empty() {
            let ptr = iter.as_ptr();
            if visited.contains(&ptr) {
                if root {
                    if let Some(group) = groups.last_mut() {
                        group.push(idx);
                    }
                }
                continue;
            } else {
                if root {
                    groups.push(vec![idx]);
                }
                visited.insert(ptr);
            }
        } else {
            let childs = iter.borrow().childs_();
            if childs.len() == 1 {
                let child = childs.get(0).unwrap();
                if visited.contains(&child.as_ptr()) {
                    if root {
                        if let Some(group) = groups.last_mut() {
                            group.push(idx);
                        }
                    }
                    continue;
                } else {
                    if root {
                        groups.push(vec![idx]);
                    }
                    visited.insert(child.as_ptr());
                }
            } else {
                assert!(childs.len() == 2, "only support fuse");
                if root {
                    groups.push(vec![idx]);
                }
            }
            pack_groups(&iter.borrow().childs_(), visited, groups, false);
        }
    }
}

pub fn fuse(shape: &mut Vec<Rc<RefCell<Iter>>>, axis1: usize, axis2: usize) {
    let mut available_axes = collect_available_axes(shape, &mut HashSet::new());
    // print_available_axes(&available_axes);
    let fused = Iter::FuseVar(FuseVar {
        lhs: available_axes[axis1].clone(),
        rhs: available_axes[axis2].clone(),
        var: Variable::new(
            format!(
                "fused_{}_{}",
                available_axes[axis1].borrow().var(),
                available_axes[axis2].borrow().var()
            )
        ),
        start: (0).into(),
        end: available_axes[axis2].borrow().end() * available_axes[axis1].borrow().end(),
        step: (1).into(),
        childs: vec![].into(),
    });
    let fused = Rc::new(RefCell::new(fused));
    available_axes.get_mut(axis1).unwrap().borrow_mut().push_child(fused.clone());
    available_axes.get_mut(axis2).unwrap().borrow_mut().push_child(fused.clone());
}

pub fn split(shape: &mut Vec<Rc<RefCell<Iter>>>, axis: usize, factor: PrimeExpr) {
    let mut available_axes = collect_available_axes(shape, &mut HashSet::new());
    // print_available_axes(&available_axes);
    let (outer, inner) = {
        let parent = available_axes[axis].borrow();
        match &*parent {
            Iter::IterVar(var) => {
                let end = (&var.end + (&factor - 1)).floor_div(&factor);
                let outer = Iter::IterVar(
                    IterVar::make(
                        format!("{}_outer", var.var),
                        &var.start,
                        end,
                        1,
                        Some(available_axes[axis].clone())
                    )
                );
                let inner = Iter::IterVar(
                    IterVar::make(
                        format!("{}_inner", var.var),
                        0,
                        factor,
                        1,
                        Some(available_axes[axis].clone())
                    )
                );
                (outer, inner)
            }
            Iter::FuseVar(var) => {
                let end = (&var.end + (&factor - 1)).floor_div(&factor);
                let outer = Iter::IterVar(
                    IterVar::make(
                        format!("{}_outer", var.var),
                        &var.start,
                        end,
                        1,
                        Some(available_axes[axis].clone())
                    )
                );
                let inner = Iter::IterVar(
                    IterVar::make(
                        format!("{}_inner", var.var),
                        0,
                        factor,
                        1,
                        Some(available_axes[axis].clone())
                    )
                );
                (outer, inner)
            }
        }
    };
    available_axes
        .get_mut(axis)
        .unwrap()
        .borrow_mut()
        .push_child(Rc::new(RefCell::new(outer)));
    available_axes
        .get_mut(axis)
        .unwrap()
        .borrow_mut()
        .push_child(Rc::new(RefCell::new(inner)));
}

pub fn print_available_axes(available_axes: &Vec<Rc<RefCell<Iter>>>) {
    print!("available_axes: [");
    for i in available_axes.iter() {
        print!("{}, ", i.borrow());
    }
    println!("]");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::halide::variable::Variable;

    #[test]
    fn test_iter_var() {
        let i = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("i", 0, 10, 1, None))));
        let j = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("j", 0, 10, 1, None))));
        let k = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("k", 0, 10, 1, None))));
        let l = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("l", 0, 10, 1, None))));
        let m = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("m", 0, 10, 1, None))));
        let mut shape = vec![i, j, k, l, m];
        fuse(&mut shape, 1, 2);
        fuse(&mut shape, 1, 2);
        fuse(&mut shape, 1, 2);
        fuse(&mut shape, 0, 1);
    }

    #[test]
    fn test_split() {
        let i = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("i", 0, 10, 1, None))));
        let j = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("j", 0, 10, 1, None))));
        let k = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("k", 0, 10, 1, None))));
        let l = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("l", 0, 10, 1, None))));
        let m = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("m", 0, 10, 1, None))));
        let mut shape = vec![i, j, k, l, m];
        split(&mut shape, 1, (2).into());
        split(&mut shape, 1, (2).into());
        split(&mut shape, 1, (2).into());
    }

    #[test]
    fn test_mix() {
        let i = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("i", 0, 10, 1, None))));
        let j = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("j", 0, 10, 1, None))));
        let k = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("k", 0, 10, 1, None))));
        let l = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("l", 0, 10, 1, None))));
        let m = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("m", 0, 10, 1, None))));
        let mut shape = vec![i, j, k, l, m];
        fuse(&mut shape, 0, 1);
        fuse(&mut shape, 0, 1);
        split(&mut shape, 0, (32).into());
        print_available_axes(&collect_available_axes(&shape, &mut HashSet::new()));
    }
}
