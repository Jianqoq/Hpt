use crate::te::{context::Context, tensor::Tensor};



impl Context {
    pub fn softmax(&mut self, a: &Tensor, axis: i64) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        todo!()
    }
}