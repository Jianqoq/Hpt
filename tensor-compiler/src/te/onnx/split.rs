use crate::te::{ context::Context, tensor::Tensor };

impl Context {
    #[track_caller]
    pub fn split(&mut self, a: &Tensor) -> Vec<Tensor> {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        todo!()
    }
}
