use std::{ collections::VecDeque, sync::Arc };

use hashbrown::HashMap;

use crate::{
    halide::{
        prime_expr::PrimeExpr,
        printer::IRPrinter,
        traits::{ AccepterMut, AccepterMutate },
        variable::Variable,
    },
    hlir::tensor::Tensor,
    to_prim_expr::ToPrimeExpr,
};
use crate::halide::traits::MutatorGetSet;

use super::{ lowered::SubstituteLoad, temp::Temp, transforms::Transforms };

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

    pub fn split(
        &mut self,
        tensor: &Tensor,
        axis: impl Into<PrimeExpr>,
        inner_loop_size: impl Into<PrimeExpr>
    ) {
        let inner_loop_size = inner_loop_size.into();
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Split(axis.into(), inner_loop_size));
        } else {
            panic!("Schedule::split: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.clone());
    }

    pub fn fuse(&mut self, tensor: &Tensor, axes: &[&dyn ToPrimeExpr]) {
        let axes = axes
            .iter()
            .map(|x| x.to_prime_expr())
            .collect::<Vec<PrimeExpr>>();
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Fuse(axes));
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

    pub fn reorder(&mut self, tensor: &Tensor, axes: &[&dyn ToPrimeExpr]) {
        let axes = axes
            .iter()
            .map(|x| x.to_prime_expr())
            .collect::<Vec<PrimeExpr>>();
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::Reorder(axes));
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
    pub fn lower(&mut self) {
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
                        let to_inline = ret
                            .get(&name)
                            .expect("Schedule::lower: target does not exist in ret");
                        let inputs = ret
                            .get(t.name_())
                            .expect("Schedule::lower: to_inline does not exist in ret")
                            .inputs.clone();
                        let body = ret
                            .get(t.name_())
                            .expect("Schedule::lower: to_inline does not exist in ret")
                            .body.clone();

                        // assume C[c0, c1] = A[c0, c1] + A[c1, c0], target = D[d0, d1] = C[d0, d1] + C[d1, d0]
                        let mut subs_load = SubstituteLoad::new(
                            Variable::make(&name),
                            Arc::new(
                                to_inline.shape
                                    .iter()
                                    .map(|x| x.var().into())
                                    .collect()
                            ),
                            to_inline.body.clone()
                        );
                        for target_input in inputs.iter() {
                            if target_input.name().name == name {
                                let target_indices = &target_input.dims; // [d0, d1]
                                subs_load.insert(target_indices.clone()); // {c0: d0, c1: d1} or {c1: d0, c0: d1}
                            }
                        }
                        body.accept_mutate(&mut subs_load);
                        let new_body = subs_load.expr().clone();
                        ret.get_mut(t.name_()).unwrap().body = new_body;
                    }
                    Transforms::Split(axis, inner_loop_size) => {}
                    Transforms::Fuse(axes) => {}
                    Transforms::ComputeAt(target, axis) => {}
                    Transforms::Reorder(axes) => {}
                    Transforms::Tile(axes) => {}
                }
            } else {
                panic!("Schedule::lower: temp does not exist in temps_map");
            }
        }
        for (t, tmp) in ret.iter() {
            println!("{}: {}", t, tmp.body);
        }
    }
}

#[cfg(test)]
mod tests {
    use tensor_types::dtype::Dtype;

    use crate::{ halide::variable::Variable, hlir::tensor::compute };

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
}
