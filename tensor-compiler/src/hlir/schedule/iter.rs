#![allow(dead_code)]
#![allow(unused_imports)]

use std::{ cell::RefCell, fmt::{ Debug, Display }, rc::Rc, sync::Arc };

use hashbrown::HashSet;
use tensor_types::dtype::Dtype;

use crate::halide::{ exprs::Int, prime_expr::PrimeExpr, variable::Variable };
#[derive(Clone)]
pub struct IterVar {
    pub(crate) var: Variable,
    pub(crate) start: PrimeExpr,
    pub(crate) end: PrimeExpr,
    pub(crate) step: PrimeExpr,
    pub(crate) childs: Arc<Vec<Rc<RefCell<Iter>>>>,
}

impl IterVar {
    pub fn make<A: Into<Variable>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<PrimeExpr>>(
        name: A,
        start: B,
        end: C,
        step: D
    ) -> Self {
        IterVar {
            var: name.into(),
            start: start.into(),
            end: end.into(),
            step: step.into(),
            childs: vec![].into(),
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
    print_available_axes(&available_axes);
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
    print_available_axes(&available_axes);
    let (outer, inner) = {
        let parent = available_axes[axis].borrow();
        match &*parent {
            Iter::IterVar(var) => {
                let end = (&var.end + (&factor - 1)).floor_div(&factor);
                let outer = Iter::IterVar(
                    IterVar::make(format!("{}_outer", var.var), &var.start, end, 1)
                );
                let inner = Iter::IterVar(
                    IterVar::make(format!("{}_inner", var.var), 0, factor, 1)
                );
                (outer, inner)
            }
            Iter::FuseVar(var) => {
                let end = (&var.end + (&factor - 1)).floor_div(&factor);
                let outer = Iter::IterVar(
                    IterVar::make(format!("{}_outer", var.var), &var.start, end, 1)
                );
                let inner = Iter::IterVar(
                    IterVar::make(format!("{}_inner", var.var), 0, factor, 1)
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
        let i = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("i", 0, 10, 1))));
        let j = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("j", 0, 10, 1))));
        let k = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("k", 0, 10, 1))));
        let l = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("l", 0, 10, 1))));
        let m = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("m", 0, 10, 1))));
        let mut shape = vec![i, j, k, l, m];
        fuse(&mut shape, 1, 2);
        fuse(&mut shape, 1, 2);
        fuse(&mut shape, 1, 2);
        fuse(&mut shape, 0, 1);
    }

    #[test]
    fn test_split() {
        let i = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("i", 0, 10, 1))));
        let j = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("j", 0, 10, 1))));
        let k = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("k", 0, 10, 1))));
        let l = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("l", 0, 10, 1))));
        let m = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("m", 0, 10, 1))));
        let mut shape = vec![i, j, k, l, m];
        split(&mut shape, 1, (2).into());
        split(&mut shape, 1, (2).into());
        split(&mut shape, 1, (2).into());
    }

    #[test]
    fn test_mix() {
        let i = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("i", 0, 10, 1))));
        let j = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("j", 0, 10, 1))));
        let k = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("k", 0, 10, 1))));
        let l = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("l", 0, 10, 1))));
        let m = Rc::new(RefCell::new(Iter::IterVar(IterVar::make("m", 0, 10, 1))));
        let mut shape = vec![i, j, k, l, m];
        fuse(&mut shape, 0, 1);
        fuse(&mut shape, 0, 1);
        split(&mut shape, 0, (32).into());
        print_available_axes(&collect_available_axes(&shape, &mut HashSet::new()));
    }
}
