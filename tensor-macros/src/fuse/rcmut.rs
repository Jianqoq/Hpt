use std::{cell::RefCell, rc::Rc};



pub(crate) struct RCMut<T> {
    pub(crate) ssa_ctx: Rc<RefCell<T>>,
}

impl<T> RCMut<T> {
    pub(crate) fn new(value: T) -> Self {
        Self { ssa_ctx: Rc::new(RefCell::new(value)) }
    }
    pub(crate) fn borrow(&self) -> std::cell::Ref<'_, T> {
        self.ssa_ctx.borrow()
    }
    pub(crate) fn borrow_mut(&self) -> std::cell::RefMut<'_, T> {
        self.ssa_ctx.borrow_mut()
    }
}

impl<T> Clone for RCMut<T> {
    fn clone(&self) -> Self {
        Self { ssa_ctx: self.ssa_ctx.clone() }
    }
}
