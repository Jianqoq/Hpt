use hashbrown::HashMap;

use crate::hlir::tensor::Tensor;

use super::temp::Temp;

pub struct Schedule {
    pub temps_map: HashMap<Tensor, Temp>,
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            temps_map: HashMap::new(),
        }
    }

    pub fn create(tensors: &[&Tensor]) -> Self {
        let mut schedule = Self::new();
        for tensor in tensors {
            schedule.temps_map.insert((*tensor).clone(), Temp::from(*tensor));
        }
        schedule
    }
}
