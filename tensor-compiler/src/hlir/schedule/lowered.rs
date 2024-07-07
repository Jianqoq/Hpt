use std::ops::{ Deref, DerefMut };

use crate::{ halide::traits::IRMutVisitor, hlir::tensor_slice::TensorSlice };

pub struct FindInputs {
    vec: Vec<TensorSlice>,
}

impl FindInputs {
    pub fn new() -> Self {
        FindInputs { vec: Vec::new() }
    }
    pub fn to_vec(self) -> Vec<TensorSlice> {
        self.vec
    }
}

impl Deref for FindInputs {
    type Target = Vec<TensorSlice>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl DerefMut for FindInputs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl IRMutVisitor for FindInputs {
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) {
        self.vec.push(slice.clone());
    }
}
