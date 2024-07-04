use std::{ collections::VecDeque, sync::Arc };

use hashbrown::HashMap;

use crate::{ halide::prime_expr::PrimeExpr, hlir::tensor::Tensor, to_prim_expr::ToPrimeExpr };

use super::{ temp::Temp, transforms::Transforms };

pub struct Schedule {
    pub temps_map: HashMap<Tensor, Temp>,
    pub records: VecDeque<Arc<String>>,
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
        let target = self.temps_map.get_mut(target);
        if let Some(temp) = target {
            if temp.inputs.contains(to_inline) {
                temp.transforms.push_back(Transforms::ComputeInline(to_inline.name_().clone()));
                temp.inputs.retain(|x| x != to_inline);
            } else {
                panic!("Schedule::inline: target does not contain to_inline");
            }
        } else {
            panic!("Schedule::inline: to_inline does not exist in temps_map");
        }
        self.records.push_back(to_inline.name_().clone());
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
        self.records.push_back(tensor.name_().clone());
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
        self.records.push_back(tensor.name_().clone());
    }

    pub fn compute_at(&mut self, tensor: &Tensor, target: &Tensor, axis: impl Into<PrimeExpr>) {
        let temp = self.temps_map.get_mut(tensor);
        if let Some(temp) = temp {
            temp.transforms.push_back(Transforms::ComputeAt(target.name_().clone(), axis.into()));
        } else {
            panic!("Schedule::compute_at: tensor does not exist in temps_map");
        }
        self.records.push_back(tensor.name_().clone());
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
        self.records.push_back(tensor.name_().clone());
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
        self.records.push_back(tensor.name_().clone());
    }
}

#[cfg(test)]
mod tests {
    use tensor_types::dtype::Dtype;

    use super::*;

    #[test]
    fn test_schedule_inline() {
        let a = Tensor::placeholder(&[&1i32, &2i32, &3i32], Dtype::I64, "a");
        let b = Tensor::placeholder(&[&1i32, &2i32, &3i32], Dtype::I64, "b");

        let add = a.add(&b);
        let sub = add.sub(&b);
        let div = sub.div(&b);

        let mut schedule = Schedule::create(&[&a, &b, &sub, &div]);
        schedule.inline(&sub, &div);
        schedule.fuse(&a, &[&1i32, &2i64]);
    }
}
