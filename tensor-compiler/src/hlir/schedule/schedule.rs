#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use std::collections::VecDeque;
use std::sync::Arc;
use crate::halide::substitute::subsititue_expr::SubstituteExpr;
use crate::halide::traits::{ AccepterMutate, MutatorGetSet };
use hashbrown::HashMap;
use crate::{ halide::prime_expr::PrimeExpr, hlir::tensor::Tensor, to_prim_expr::ToPrimeExpr };
use crate::hlir::schedule::iter::gen_indices;
use super::iter::fuse;
use super::{ iter::split, temp::Temp, transforms::Transforms };

pub struct Schedule {
    pub temps_map: HashMap<Tensor, Temp>,
    pub records: VecDeque<Tensor>,
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            temps_map: HashMap::new(),
            records: VecDeque::new(),
        }
    }

    pub fn create(tensors: &[&Tensor]) -> Self {
        let mut schedule = Self::new();
        for tensor in tensors {
            let temp = Temp::from(*tensor);
            schedule.temps_map.insert((*tensor).clone(), temp);
        }
        schedule
    }

    /// to_inline: C = A + B, target: D = C + E, inline C to D, then D = (A + B) + E
    pub fn inline(&mut self, to_inline: &Tensor, target: &Tensor) {
        if let Some(temp) = self.temps_map.get_mut(target) {
            let contain = temp.inputs
                .iter()
                .map(|x| &x.name().name == to_inline.name_())
                .any(|x| x);
            if contain {
                temp.transforms.push_back(Transforms::Inline(to_inline.name_().clone()));
            } else {
                panic!("Schedule::inline: target does not contain to_inline");
            }
        } else {
            panic!("Schedule::inline: to_inline does not exist in temps_map");
        }
        self.records.push_back(target.clone());
    }

    pub fn split(&mut self, tensor: &Tensor, axis: usize, inner_loop_size: impl Into<PrimeExpr>) {
        let inner_loop_size = inner_loop_size.into();
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Split(temp.name.clone(), axis, inner_loop_size));
        } else {
            panic!("Schedule::split: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.clone());
    }

    pub fn fuse(&mut self, tensor: &Tensor, axis1: usize, axis2: usize) {
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Fuse(temp.name.clone(), axis1, axis2));
        } else {
            panic!("Schedule::fuse: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.clone());
    }

    pub fn compute_at(&mut self, tensor: &Tensor, target: &Tensor, axis: impl Into<PrimeExpr>) {
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::ComputeAt(target.name_().clone(), axis.into()));
        } else {
            panic!("Schedule::compute_at: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.clone());
    }

    pub fn reorder(&mut self, tensor: &Tensor, axes: &[usize]) {
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Reorder(temp.name.clone(), axes.to_vec()));
        } else {
            panic!("Schedule::reorder: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.clone());
    }
    pub fn tile(&mut self, tensor: &Tensor, axes: &[&dyn ToPrimeExpr]) {
        assert!(axes.len() == 2, "Schedule::tile: axes length must be 2");
        let axes = axes
            .iter()
            .map(|x| x.to_prime_expr())
            .collect::<Vec<PrimeExpr>>();
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Tile(axes));
        } else {
            panic!("Schedule::tile: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.clone());
    }
    pub fn lower(&mut self) -> HashMap<Arc<String>, Temp> {
        let mut ret = HashMap::new();

        for (t, tmp) in self.temps_map.iter() {
            ret.insert(t.name_().clone(), tmp.clone());
        }

        while let Some(t) = self.records.pop_front() {
            let temp = self.temps_map.get_mut(&t);
            if let Some(temp) = temp {
                let transform = temp.transforms
                    .pop_front()
                    .expect("Schedule::lower: transform is empty");
                match transform {
                    Transforms::Inline(name) => {
                        if let Some(to_inline) = ret.get(&name) {
                        }
                    }
                    Transforms::Split(name, axis, inner_loop_size) => {
                        if let Some(_temp) = ret.get_mut(&name) {
                            split(
                                &mut _temp.shape,
                                &_temp.saved_shape_order,
                                axis,
                                inner_loop_size
                            );
                            let indices = gen_indices(&_temp.shape);
                            let mut subs_expr = SubstituteExpr::new();
                            for (origin, new) in temp.shape.iter().zip(indices.iter()) {
                                subs_expr.add_replacement(
                                    origin.borrow().var().to_prime_expr(),
                                    new.clone()
                                );
                            }
                            temp.body.accept_mutate(&mut subs_expr);
                            _temp.body = subs_expr.expr().clone();
                            // update saved_shape_order
                            let mut vec = vec![];
                            for i in _temp.saved_shape_order.iter() {
                                if *i < axis {
                                    vec.push(*i);
                                } else if *i == axis {
                                    vec.push(*i);
                                    vec.push(*i + 1);
                                } else {
                                    vec.push(*i + 1);
                                }
                            }
                            _temp.saved_shape_order = vec;
                        }
                    }
                    Transforms::Fuse(name, axis1, axis2) => {
                        if let Some(_temp) = ret.get_mut(&name) {
                            fuse(&mut _temp.shape, &_temp.saved_shape_order, axis1, axis2);
                            let indices = gen_indices(&_temp.shape);
                            let mut subs_expr = SubstituteExpr::new();
                            for (origin, new) in temp.shape.iter().zip(indices.iter()) {
                                subs_expr.add_replacement(
                                    origin.borrow().var().to_prime_expr(),
                                    new.clone()
                                );
                            }
                            temp.body.accept_mutate(&mut subs_expr);
                            _temp.body = subs_expr.expr().clone();

                            // update saved_shape_order
                            let mut vec = vec![];
                            for i in _temp.saved_shape_order.iter() {
                                if *i < axis2 {
                                    vec.push(*i);
                                } else if *i == axis2 {
                                    continue;
                                } else {
                                    vec.push(*i - 1);
                                }
                            }
                            _temp.saved_shape_order = vec;
                        }
                    }
                    Transforms::ComputeAt(target, axis) => {}
                    Transforms::Reorder(name, new_order) => {
                        if let Some(_temp) = ret.get_mut(&name) {
                            assert!(
                                new_order.len() == _temp.saved_shape_order.len(),
                                "Schedule::lower: new_order length must be equal to shape length"
                            );
                            _temp.saved_shape_order = new_order.clone();
                        }
                    }
                    Transforms::Tile(axes) => {}
                }
            } else {
                panic!("Schedule::lower: temp does not exist in temps_map");
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use tensor_types::dtype::Dtype;

    use crate::{
        halide::variable::Variable,
        hlir::{ schedule::iter::gen_indices, tensor::compute },
    };

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

        let mut schedule = Schedule::create(&[&a, &c, &d]);
        schedule.inline(&c, &d);
        schedule.lower();
    }

    #[test]
    fn test_schedule_split() {
        let m = Variable::make("m");
        let n = Variable::make("n");
        let o = Variable::make("o");
        let p = Variable::make("p");

        let a = Tensor::placeholder(&[&m, &n, &o, &p], Dtype::I64, "A");

        let c = compute(Dtype::BF16, [&n, &m, &o, &p], [&a], "C", |[a], [i, j, k, l]| {
            a.slice([&i, &j, &k, &l])
        });

        println!("===================== testing fuse [[m, n], [o, p]] =====================");
        let mut schedule = Schedule::create(&[&a, &c]);
        schedule.fuse(&c, 0, 1);
        schedule.fuse(&c, 1, 2);
        schedule.fuse(&c, 0, 1);
        schedule.lower();
        println!("===================== testing split [[m], [n], [o], [p]] =====================");
        let mut schedule = Schedule::create(&[&a, &c]);
        schedule.split(&c, 0, 32);
        schedule.split(&c, 1, 32);
        schedule.split(&c, 2, 32);
        schedule.split(&c, 3, 32);
        schedule.lower();
    }

    #[test]
    fn test_fuse_split() {
        let m = Variable::make("m");
        let n = Variable::make("n");
        let o = Variable::make("o");
        let p = Variable::make("p");

        let a = Tensor::placeholder(&[&m, &n, &o, &p], Dtype::I64, "A");

        let c = compute(Dtype::BF16, [&n, &m, &o, &p], [&a], "C", |[a], [i, j, k, l]| {
            a.slice([&i + 4, j, k, l])
        });

        let mut schedule = Schedule::create(&[&a, &c]);
        schedule.fuse(&c, 0, 1);
        schedule.split(&c, 1, 16);
        schedule.fuse(&c, 0, 1);
        schedule.split(&c, 0, 16);
        // schedule.split(&c, 3, 32);
        // schedule.split(&c, 3, 48);
        // schedule.split(&c, 5, 64);
        // schedule.fuse(&c, 2, 3);
        let map = schedule.lower();
        println!("{}", map[c.name_()].body);
    }

    #[test]
    fn test_fuse_split2() {
        let m = Variable::make("m");
        let n = Variable::make("n");

        let a = Tensor::placeholder(&[&m, &n], Dtype::I64, "A");

        let c = compute(Dtype::BF16, [&n, &m], [&a], "C", |[a], [i, j]| { a.slice([&i, &j]) });

        let mut schedule = Schedule::create(&[&a, &c]);
        schedule.fuse(&c, 0, 1);
        schedule.split(&c, 0, 16);
        schedule.lower();
    }

    #[test]
    fn test_reorder_split() {
        let m = Variable::make("m");
        let n = Variable::make("n");

        let a = Tensor::placeholder(&[&m], Dtype::I64, "A");

        let c = compute(Dtype::BF16, [&n, &m], [&a], "C", |[a], [i, _]| { a.slice([i]) });

        let mut schedule = Schedule::create(&[&a, &c]);
        schedule.split(&c, 0, 16);
        schedule.reorder(&c, &[1, 0, 2]);
        schedule.split(&c, 0, 7);
        schedule.fuse(&c, 0, 1);
        let map = schedule.lower();
        println!("{}", map[c.name_()].body);
    }
}
