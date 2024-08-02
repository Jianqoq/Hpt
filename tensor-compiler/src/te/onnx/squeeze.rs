use crate::te::{context::Context, tensor::Tensor};

impl Context {
    #[track_caller]
    pub fn squeeze(&mut self, a: &Tensor) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        todo!()
    }
}